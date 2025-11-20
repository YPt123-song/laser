import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
import threading
import queue
import os
import sys

# ================= 配置参数 =================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

THRESH_MIN = 90       
CIRCULARITY_PARAM = 0.8 # 圆形度 (宽松)
AREA_MIN = 100         
SEND_INTERVAL = 0.05   # 20Hz 控制频率

# [新增] 闲置归位时间 (秒)
# 如果超过这个时间没有检测到圆，云台自动回中 (0,0)
IDLE_TIME = 5.0       

# ================= 串口工作线程 =================
class SerialWorker(threading.Thread):
    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self.ser = None
        self.running = True
        self.connected = False
        self.send_queue = queue.Queue() # 发送队列
        self.daemon = True # 设置为守护线程，主程序退出时自动结束

    def connect(self):
        try:
            if self.ser:
                self.ser.close()
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.01,       # 读超时非常短
                write_timeout=0.1,  # 写超时
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            self.ser.dtr = False
            self.ser.rts = False
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.connected = True
            print(f"[Serial] 成功连接 {self.port}")
        except Exception as e:
            self.connected = False
            # print(f"[Serial] 连接失败: {e}") # 减少刷屏

    def send(self, data_type, values):
        """发送标准视觉数据 (以 & 结尾)"""
        if not self.connected: return
        
        # 构建协议帧: $target,320/240&
        data_str = "/".join(str(v) for v in values)
        msg = f"${data_type},{data_str}&"
        self._enqueue(msg)

    def send_raw(self, msg):
        """[新增] 发送原始指令 (用于发送以 # 或 * 结尾的指令)"""
        if not self.connected: return
        self._enqueue(msg)

    def _enqueue(self, msg):
        # 如果队列堆积太多(说明发不过来)，清空旧的，只发最新的
        if self.send_queue.qsize() > 2:
            try:
                with self.send_queue.mutex:
                    self.send_queue.queue.clear()
            except:
                pass
        self.send_queue.put(msg)

    def run(self):
        while self.running:
            if not self.connected:
                self.connect()
                time.sleep(1.0) # 重连间隔
                continue

            try:
                # 1. 接收 (非阻塞)
                if self.ser.in_waiting:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            print(f"[HW] {line}")
                    except Exception:
                        pass

                # 2. 发送
                if not self.send_queue.empty():
                    msg = self.send_queue.get_nowait()
                    self.ser.write(msg.encode('ascii'))
                    # self.ser.flush() # 根据情况，有时 flush 会阻塞，如果卡顿可注释掉

            except Exception as e:
                print(f"[Serial] 通信异常: {e}")
                self.connected = False
                time.sleep(0.1)
            
            # 让出 CPU
            time.sleep(0.005) 

    def stop(self):
        self.running = False
        if self.ser: self.ser.close()

# ================= 摄像头工作线程 =================
class CameraWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.daemon = True
        self.cam_idx = -1
        self.cap = None
        self.fps = 0
        self._last_frame_time = time.time()

    def find_valid_camera(self):
        # 优先扫描常用 USB 摄像头索引
        check_list = [0, 1, 11, 21, 2, 3, 4, 12, 22]
        if sys.platform.startswith('linux'):
            os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

        for index in check_list:
            path = f"/dev/video{index}"
            if not os.path.exists(path): continue

            temp_cap = cv2.VideoCapture(index)
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # 低分辨率测通
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            if temp_cap.isOpened():
                # 连续读几帧确保稳定
                ok_count = 0
                for _ in range(3):
                    ret, _ = temp_cap.read()
                    if ret: ok_count += 1
                    time.sleep(0.05)
                
                if ok_count >= 2:
                    print(f"[Camera] 锁定摄像头索引: {index}")
                    temp_cap.release()
                    return index
            temp_cap.release()
        return -1

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def run(self):
        while self.running:
            # 1. 如果未连接，尝试连接
            if self.cap is None or not self.cap.isOpened():
                idx = self.find_valid_camera()
                if idx != -1:
                    self.cap = cv2.VideoCapture(idx)
                    # 设定工作分辨率
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    # 设定 MJPG 格式 (香橙派上通常更快)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                else:
                    time.sleep(1.0)
                    continue

            # 2. 读取帧
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                
                # 计算采集 FPS
                now = time.time()
                self.fps = 1.0 / (now - self._last_frame_time) if (now - self._last_frame_time) > 0 else 0
                self._last_frame_time = now
            else:
                print("[Camera] 读取失败或掉线")
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
            
            # 摄像头线程不需要 sleep，全速跑

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()

# ================= 主程序 (图像处理) =================
def main():
    # 启动工作线程
    cam_thread = CameraWorker()
    ser_thread = SerialWorker(SERIAL_PORT, BAUD_RATE)
    
    cam_thread.start()
    ser_thread.start()

    print(f"[System] 启动多线程模式...")
    print(f"[System] 阈值: {THRESH_MIN}, 圆形度: {CIRCULARITY_PARAM}, 闲置超时: {IDLE_TIME}s")
    
    last_send_time = 0
    process_fps = 0
    process_time_start = time.time()

    # [新增] 闲置状态追踪
    last_detection_time = time.time()
    is_idle_mode = False

    try:
        while True:
            # 1. 从线程获取最新帧 (非阻塞)
            frame = cam_thread.get_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue

            # 2. 图像处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, THRESH_MIN, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_target = None
            max_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < AREA_MIN: continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * area / (perimeter ** 2)

                if circularity > CIRCULARITY_PARAM:
                    if area > max_area:
                        max_area = area
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        best_target = (int(x), int(y), int(radius), circularity)
                else:
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 1)

            # 3. 逻辑判断：追踪 vs 归位
            current_time = time.time()

            if best_target:
                # 发现目标：更新时间，退出闲置模式
                last_detection_time = current_time
                is_idle_mode = False

                cx, cy, r, circ = best_target
                # 绘制
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 3)
                text_info = f"({cx},{cy}) {circ:.2f}"
                cv2.putText(frame, text_info, (cx - 20, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 发送追踪指令
                if current_time - last_send_time > SEND_INTERVAL:
                    ser_thread.send("target", [cx, cy])
                    last_send_time = current_time
            else:
                # 未发现目标：检查时间
                if not is_idle_mode and (current_time - last_detection_time > IDLE_TIME):
                    print(f"[System] {IDLE_TIME}秒未检测到目标，自动归中 (0,0)")
                    # 发送归位指令 (使用原始格式 $Angle_x,val#)
                    ser_thread.send_raw("$Angle_0,0.0#")
                    ser_thread.send_raw("$Angle_1,0.0#")
                    is_idle_mode = True
                
                if is_idle_mode:
                    cv2.putText(frame, "IDLE: Homing", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 4. 计算处理帧率
            process_fps = 1.0 / (current_time - process_time_start) if (current_time - process_time_start) > 0 else 0
            process_time_start = current_time

            # 5. 显示
            h, w = frame.shape[:2]
            display_frame = cv2.resize(frame, (w//2, h//2))
            
            status_text = "Ser:OK" if ser_thread.connected else "Ser:--"
            cv2.putText(display_frame, f"CamFPS: {int(cam_thread.fps)} ProcFPS: {int(process_fps)}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, status_text, 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Multithreaded Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[System] 正在停止...")
    finally:
        cam_thread.stop()
        ser_thread.stop()
        cv2.destroyAllWindows()
        # 等待线程结束
        cam_thread.join(timeout=1.0)
        ser_thread.join(timeout=1.0)
        print("[System] 退出完成")

if __name__ == "__main__":
    main()