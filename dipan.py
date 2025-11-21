#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手柄（evdev） + 串口 控制麦轮小车
版本：V5.4 (连贯惯性漂移版)

操作方式：
1. 右扳机加速到200以上
2. 按住TR进入漂移状态
3. 右摇杆控制漂移方向和强度
4. 松开TR平滑恢复正常控制
"""

import time
import threading
import serial
import math
import sys
from evdev import InputDevice, ecodes

# ==================== 参数配置区 ====================
BASE_SPEED    = 150
MAX_SPEED     = 300
MIN_SPEED     = 50
ANGULAR_SPEED = 100

# --- 漂移参数 ---
DRIFT_SPEED_THRESHOLD = 200   # 触发漂移最低速度
DRIFT_REAR_FACTOR     = -0.6  # 后轮系数(负=反转)
DRIFT_FRONT_BOOST     = 1.2   # 前轮增益
DRIFT_SPIN_GAIN       = 180   # 右摇杆->旋转增益(越大越灵敏)
DRIFT_FRICTION        = 0.985 # 惯性衰减系数(越接近1惯性越强)
DRIFT_MIN_SPEED       = 80    # 漂移中最低保持速度

# --- 运动参数 ---
ACCEL_STEP    = 25
DECEL_STEP    = 60
CONTROL_HZ    = 60

SERIAL_PORT   = "/dev/ttyUSB0"
BAUDRATE      = 115200
TIMEOUT       = 0.1
TARGET_DEVICE = "/dev/input/event6"

LEFT_FACTOR   = 1.0
RIGHT_FACTOR  = 1.0
# ===================================================

def inverse_kinematics(vx, vy, omega, drift_info=None):
    """
    drift_info: None或dict{active, direction, intensity}
    """
    lf =  vx - vy - omega
    lr =  vx + vy - omega
    rf =  vx + vy + omega
    rr =  vx - vy + omega

    if drift_info and drift_info['active']:
        intensity = drift_info['intensity']  # 0~1
        direction = drift_info['direction']  # -1或1
        
        # 后轮：根据强度插值(正常->反转)
        rear_factor = 1.0 + (DRIFT_REAR_FACTOR - 1.0) * intensity
        lr *= rear_factor
        rr *= rear_factor
        
        # 前轮：轻微增强
        front_factor = 1.0 + (DRIFT_FRONT_BOOST - 1.0) * intensity
        lf *= front_factor
        rf *= front_factor
        
        # 差速旋转
        spin = DRIFT_SPIN_GAIN * intensity * direction
        lf -= spin
        lr -= spin
        rf += spin
        rr += spin

    lf *= LEFT_FACTOR
    lr *= LEFT_FACTOR
    rf *= RIGHT_FACTOR
    rr *= RIGHT_FACTOR
    return [lf, lr, rf, rr]

class SerialPort:
    def __init__(self, port, baud, timeout):
        try:
            self.ser = serial.Serial(port, baud, timeout=timeout)
            if not self.ser.is_open:
                self.ser.open()
            print(f"[INFO] 串口已打开: {port} @ {baud}")
        except Exception as e:
            print(f"[FATAL] 串口打开失败: {e}")
            sys.exit(1)

    def send(self, txt):
        if not txt.endswith('#'):
            txt += '#'
        try:
            self.ser.write(txt.encode('utf-8'))
        except:
            pass

    def recv(self):
        try:
            return self.ser.readline().decode('utf-8', errors='ignore').strip()
        except:
            return None

    def close(self):
        self.ser.close()

class JoystickController:
    AXIS_MAP = {
        'ABS_Y'     : ecodes.ABS_Y,
        'ABS_X'     : ecodes.ABS_X,
        'ABS_RX'    : ecodes.ABS_Z,
        'ABS_GAS'   : ecodes.ABS_GAS,
        'ABS_BRAKE' : ecodes.ABS_BRAKE,
    }
    BTN_TR = 311

    def __init__(self, ser):
        self.ser = ser
        self.device = self._connect_device()
        if self.device is None:
            sys.exit(f"❌ 无法连接 {TARGET_DEVICE}")

        print(f"✅ 手柄就绪")
        print(f"🎮 操作: R2加速 | TR+右摇杆漂移 | 速度>{DRIFT_SPEED_THRESHOLD}触发")

        # 输入状态
        self.raw_vx = 0.0
        self.raw_vy = 0.0
        self.raw_omega = 0.0
        self.trigger_left = 0
        self.trigger_right = 0

        # 运动状态(带惯性)
        self.vel_vx = 0.0
        self.vel_vy = 0.0
        self.vel_omega = 0.0

        # 漂移状态
        self.drift_pressed = False
        self.drift_active = False
        self.drift_direction = 0
        self.drift_intensity = 0.0

        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._control_loop, daemon=True).start()

    def _connect_device(self):
        try:
            return InputDevice(TARGET_DEVICE)
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def _get_speed_limit(self):
        limit = BASE_SPEED
        if self.trigger_right > 10:
            limit = BASE_SPEED + (MAX_SPEED - BASE_SPEED) * (self.trigger_right / 255.0)
        elif self.trigger_left > 10:
            limit = BASE_SPEED - (BASE_SPEED - MIN_SPEED) * (self.trigger_left / 255.0)
        return limit

    def _get_target_speeds(self):
        limit = self._get_speed_limit()
        vx = -limit * (self.raw_vx ** 3)
        vy = -limit * (self.raw_vy ** 3)
        omega = -ANGULAR_SPEED * (self.raw_omega ** 3)
        return vx, vy, omega

    def _smooth(self, current, target, accel=ACCEL_STEP, decel=DECEL_STEP):
        diff = target - current
        step = decel if (target == 0 or target * current < 0) else accel
        if abs(diff) < step:
            return target
        return current + step if diff > 0 else current - step

    def _control_loop(self):
        interval = 1.0 / CONTROL_HZ
        
        while self.running:
            t0 = time.time()
            
            current_speed = math.sqrt(self.vel_vx**2 + self.vel_vy**2)
            
            # ========== 漂移状态机 ==========
            if self.drift_pressed:
                if current_speed > DRIFT_SPEED_THRESHOLD or self.drift_active:
                    # 进入/保持漂移
                    if not self.drift_active:
                        self.drift_active = True
                        print(f"\n🔥 漂移启动! 速度:{int(current_speed)}")
                    
                    # 右摇杆控制: 方向和强度
                    if abs(self.raw_omega) > 0.08:
                        self.drift_direction = 1 if self.raw_omega > 0 else -1
                        self.drift_intensity = min(abs(self.raw_omega) * 1.5, 1.0)
                    else:
                        # 摇杆回中时强度缓慢降低
                        self.drift_intensity *= 0.92
                    
                    # 惯性衰减(自然减速)
                    self.vel_vx *= DRIFT_FRICTION
                    self.vel_vy *= DRIFT_FRICTION
                    
                    # 保持最低速度
                    speed_now = math.sqrt(self.vel_vx**2 + self.vel_vy**2)
                    if speed_now < DRIFT_MIN_SPEED and speed_now > 1:
                        scale = DRIFT_MIN_SPEED / speed_now
                        self.vel_vx *= scale
                        self.vel_vy *= scale
                    
                    # 右摇杆也可以微调横向
                    if abs(self.raw_omega) > 0.1:
                        side_push = self.raw_omega * 15  # 轻微横向推力
                        self.vel_vy += side_push
                    
                    self.vel_omega = 0  # 旋转由漂移函数处理
                    
                else:
                    # 速度不够，不触发
                    pass
            else:
                # 松开TR
                if self.drift_active:
                    print(f"\n✅ 漂移结束 - 恢复控制")
                    self.drift_active = False
                self.drift_intensity *= 0.85  # 快速衰减漂移强度
            
            # ========== 速度更新 ==========
            if not self.drift_active:
                # 正常模式：平滑跟随目标
                tx, ty, tw = self._get_target_speeds()
                self.vel_vx = self._smooth(self.vel_vx, tx)
                self.vel_vy = self._smooth(self.vel_vy, ty)
                self.vel_omega = self._smooth(self.vel_omega, tw)
            
            # ========== 输出 ==========
            drift_info = {
                'active': self.drift_active or self.drift_intensity > 0.05,
                'direction': self.drift_direction,
                'intensity': self.drift_intensity
            }
            
            wheels = inverse_kinematics(
                self.vel_vx, self.vel_vy, self.vel_omega,
                drift_info if drift_info['active'] else None
            )
            
            cmd = f"$spd:{int(wheels[0])},{int(wheels[1])},{int(wheels[2])},{int(wheels[3])}"
            self.ser.send(cmd)
            
            # 调试显示
            if self.drift_active:
                print(f"\r🔥 漂移 | 强度:{self.drift_intensity:.2f} 方向:{'→' if self.drift_direction>0 else '←'} 速度:{int(current_speed):3d} | {int(wheels[0]):4d} {int(wheels[1]):4d} {int(wheels[2]):4d} {int(wheels[3]):4d}", end="", flush=True)
            
            # 控频
            dt = time.time() - t0
            if interval - dt > 0:
                time.sleep(interval - dt)

    def _update_axis(self, axis, value):
        # 归一化
        if value > 255:
            norm = (value + 32768) / 65535 * 2 - 1
        else:
            norm = (value - 128) / 128.0
        if abs(norm) < 0.05:
            norm = 0.0

        if axis == self.AXIS_MAP['ABS_Y']:
            self.raw_vx = norm
        elif axis == self.AXIS_MAP['ABS_X']:
            self.raw_vy = norm
        elif axis == self.AXIS_MAP['ABS_RX']:
            self.raw_omega = norm
        elif axis == self.AXIS_MAP['ABS_BRAKE']:
            self.trigger_left = value
        elif axis == self.AXIS_MAP['ABS_GAS']:
            self.trigger_right = value

    def _handle_button(self, code, value):
        if code == self.BTN_TR:
            self.drift_pressed = (value == 1)
            if self.drift_pressed:
                spd = math.sqrt(self.vel_vx**2 + self.vel_vy**2)
                status = "✓" if spd > DRIFT_SPEED_THRESHOLD else f"✗ 需>{DRIFT_SPEED_THRESHOLD}"
                print(f"\n[TR按下] 速度:{int(spd)} {status}")

    def _read_loop(self):
        try:
            for ev in self.device.read_loop():
                if ev.type == ecodes.EV_ABS:
                    self._update_axis(ev.code, ev.value)
                elif ev.type == ecodes.EV_KEY:
                    self._handle_button(ev.code, ev.value)
        except Exception as e:
            print(f"[ERROR] {e}")
            self.running = False

def main():
    ser = SerialPort(SERIAL_PORT, BAUDRATE, TIMEOUT)
    js = JoystickController(ser)
    print("\n" + "="*55)
    print("🏎️  操作流程:")
    print("   1. R2加速到200+")
    print("   2. 按住TR")  
    print("   3. 右摇杆向左/右 → 控制漂移方向和强度")
    print("   4. 松开TR恢复正常")
    print("="*55 + "\n")
    try:
        while True:
            ser.recv()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[退出]")
        ser.send("$spd:0,0,0,0")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
