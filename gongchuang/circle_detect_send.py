import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
import glob
import os
import sys

# ================= 串口设置 =================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# ================= 视觉参数 =================
THRESH_MIN = 90       
CIRCULARITY_PARAM = 0.8 
AREA_MIN = 100        
SEND_INTERVAL = 0.05   

def build_frame(data_type, values):
    data_str = "/".join(str(v) for v in values)
    return f"${data_type},{data_str}&"

def find_valid_camera():
    """
    遍历寻找可用摄像头
    """
    print("\n[System] 正在扫描摄像头设备...")
    
    # 优先尝试常见 USB 摄像头索引
    # 注意：掉线重连后，摄像头可能会变成 video2, video3 等
    check_list = [0, 1, 11, 21, 2, 3, 4, 12, 22] 
    
    if sys.platform.startswith('linux'):
        os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

    for index in check_list:
        device_path = f"/dev/video{index}"
        if not os.path.exists(device_path):
            continue

        # print(f"  -> 测试 video{index} ... ", end="", flush=True)
        
        cap = cv2.VideoCapture(index)
        # 强制设为低分辨率测试，提高成功率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        if cap.isOpened():
            # 连续读 3 帧确保稳定
            success_count = 0
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[0] > 0:
                    success_count += 1
                    time.sleep(0.1)
                else:
                    break
            
            if success_count == 3:
                print(f" -> [成功] 锁定 video{index}")
                cap.release()
                return index
        
        cap.release()
        # print("跳过")
    
    print("[Warning] 未找到摄像头！")
    return -1

def main():
    # 1. 初始连接串口
    ser = None
    print(f"[System] 正在连接 {SERIAL_PORT} ...")
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=0.02,
            write_timeout=0.5,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        ser.dtr = False
        ser.rts = False
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"[System] 串口连接成功")
    except Exception as e:
        print(f"[Error] 串口失败: {e}")
        # 即使串口没连上，也继续运行去修摄像头

    # 2. 初始寻找摄像头
    cam_idx = find_valid_camera()
    cap = None
    if cam_idx != -1:
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[System] 参数 -> 阈值: {THRESH_MIN}, 圆形度: {CIRCULARITY_PARAM}")
    print("[System] 按 'q' 键退出")

    pTime = 0 
    last_send_time = 0 

    try:
        while True:
            # --- 串口保活与读取 ---
            if ser and ser.is_open:
                try:
                    while ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            print(f"[HW] {line}")
                except Exception:
                    pass

            # --- 摄像头保活与读取 ---
            frame = None
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[Error] 读取帧失败，尝试重置...")
                    cap.release()
                    cap = None
            
            # 如果摄像头断了，尝试重连
            if cap is None or not cap.isOpened():
                # 稍微延时，防止 CPU 满载
                # 只有当串口有数据（说明硬件活着）时才拼命找摄像头
                time.sleep(1.0) 
                print("[System] 尝试找回摄像头...")
                new_idx = find_valid_camera()
                if new_idx != -1:
                    cap = cv2.VideoCapture(new_idx)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"[System] 摄像头重连成功: video{new_idx}")
                continue

            if frame is None: continue

            # --- 视觉算法 ---
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, THRESH_MIN, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vis_frame = frame.copy()
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
                    cv2.drawContours(vis_frame, [contour], -1, (0, 0, 255), 1)

            # --- 发送控制 ---
            if best_target:
                cx, cy, r, circ = best_target
                cv2.circle(vis_frame, (cx, cy), r, (0, 255, 0), 2)
                text_info = f"({cx},{cy})"
                cv2.putText(vis_frame, text_info, (cx - 20, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if ser and ser.is_open:
                    now = time.time()
                    if now - last_send_time > SEND_INTERVAL:
                        msg = build_frame("target", [cx, cy])
                        try:
                            ser.write(msg.encode("ascii"))
                            ser.flush() 
                            last_send_time = now 
                        except Exception:
                            pass # 忽略发送错误，保持运行

            # --- 显示 ---
            h, w = frame.shape[:2]
            small_frame = cv2.resize(vis_frame, (w//2, h//2))
            cv2.putText(small_frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Detection", small_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("程序停止")
    finally:
        if ser: ser.close()
        if cap: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()