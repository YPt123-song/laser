import cv2
import numpy as np
import time
import multiprocessing as mp
from queue import Empty

# ================= 配置参数 =================
PROCESS_WIDTH = 640        # 检测时的图像宽度 (降采样以提速)
NUM_PROCESSES = 3          # 并行进程数 (RK3588 建议 3)

# --- HSV 阈值设置 (关键) ---
# H: 色相 0-180 (白色没有特定色相，所以全包)
# S: 饱和度 0-255 (白色必须低饱和度，越低越白。设为 0-60 允许一点泛白)
# V: 亮度 0-255 (白色必须高亮度。设为 150-255 允许一定程度的阴影)
HSV_LOWER = np.array([0, 0, 150])   
HSV_UPPER = np.array([180, 60, 255]) 

# 形状参数
MIN_CIRCULARITY = 0.6      # 圆形度阈值 (0.6 允许一定的倾斜/椭圆)
MIN_AREA_BASE = 100        # 在 640 宽度的图上的最小面积
# ===========================================

def detect_logic(frame, frame_id):
    """
    子进程运行的检测逻辑 (HSV版)
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. 降采样
    scale = PROCESS_WIDTH / orig_w
    new_h = int(orig_h * scale)
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, new_h), interpolation=cv2.INTER_NEAREST)

    # 2. 转换到 HSV 空间
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # 3. HSV 阈值过滤
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    
    # 4. 形态学去噪 (去除小白点噪点)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 闭运算 (填补圆内部的小黑洞)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. 轮廓检测
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # 面积过滤 (根据缩放比例调整阈值)
        if area < (MIN_AREA_BASE * scale * scale): 
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        
        # 计算圆形度: 4*pi*area / perimeter^2
        # 完美圆=1.0，正方形=0.785
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity > MIN_CIRCULARITY: 
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # 6. 坐标映射回原图
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_radius = int(radius / scale)
            orig_area = int(area / (scale * scale))
            
            circles.append({
                'xy': (orig_x, orig_y),
                'r': orig_radius,
                'area': orig_area,
                'circ': circularity
            })
            
    return frame_id, circles, mask # 返回 mask 仅用于调试显示(可选)

class CircleWorker(mp.Process):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True

    def run(self):
        while True:
            try:
                task = self.input_queue.get()
                if task is None: break
                
                frame_id, frame = task
                # 执行检测
                fid, circles, mask_small = detect_logic(frame, frame_id)
                
                # 为了在主进程显示 mask，我们需要把它传回去
                # 但为了性能，不要传太大的数据。这里 mask_small 只有 640 宽，是可以接受的。
                self.output_queue.put((fid, circles, mask_small))
            except Exception as e:
                print(f"Worker Error: {e}")

def main():
    print("===== RK3588 HSV 多进程圆检测 =====")
    print(f"HSV Lower: {HSV_LOWER}")
    print(f"HSV Upper: {HSV_UPPER}")
    
    mode = input("输入模式 (1=定时返回, 2=持续返回, 默认2): ").strip()
    mode = 1 if mode == '1' else 2
    
    input_queue = mp.Queue(maxsize=NUM_PROCESSES + 2)
    output_queue = mp.Queue(maxsize=NUM_PROCESSES + 2)
    
    workers = []
    for _ in range(NUM_PROCESSES):
        p = CircleWorker(input_queue, output_queue)
        p.start()
        workers.append(p)
        
    cap = cv2.VideoCapture(4) # 请根据实际情况修改 index
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 数据缓存
    current_circles = []
    current_mask = None
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    frame_id = 0
    
    last_report_time = time.time()
    report_interval = 5.0

    print("开始运行... 按 'q' 退出")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_id += 1
            
            # 1. 发送任务
            if not input_queue.full():
                input_queue.put((frame_id, frame.copy()))

            # 2. 接收结果
            try:
                while not output_queue.empty():
                    fid, res_circles, res_mask = output_queue.get_nowait()
                    current_circles = res_circles
                    current_mask = res_mask
            except Empty:
                pass

            # 3. 业务逻辑
            curr_time = time.time()
            
            if mode == 1:
                if curr_time - last_report_time >= report_interval:
                    coords = [c['xy'] for c in current_circles]
                    print(f"\n[模式1] {report_interval}s 报告:")
                    print(f"检测到 {len(current_circles)} 个圆: {coords}")
                    last_report_time = curr_time
                mode_text = "Mode: 1 (Timer)"
            else:
                # 模式2: 持续输出逻辑 (这里仅在控制台静默，以免刷屏，如需打印请取消注释)
                # if current_circles: print([c['xy'] for c in current_circles])
                mode_text = "Mode: 2 (Continuous)"

            # 4. 绘制显示
            vis_frame = frame
            
            for c in current_circles:
                x, y = c['xy']
                r = c['r']
                # 画外圆
                cv2.circle(vis_frame, (x, y), r, (0, 255, 0), 2)
                # 画圆心
                cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)
                # 显示坐标和圆形度
                label = f"({x},{y}) {c['circ']:.2f}"
                cv2.putText(vis_frame, label, (x - 40, y - r - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 性能信息
            frame_count += 1
            elapsed = curr_time - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = curr_time

            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(vis_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # --- 显示调试 Mask (画中画) ---
            if current_mask is not None:
                # 将 mask 转为 3通道 BGR 才能贴到 vis_frame 上
                mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
                # 缩放 mask 到合适大小 (比如 320x180)
                thumb_h = 180
                thumb_w = int(thumb_h * (mask_bgr.shape[1] / mask_bgr.shape[0]))
                mask_thumb = cv2.resize(mask_bgr, (thumb_w, thumb_h))
                
                # 贴到右下角
                h, w = vis_frame.shape[:2]
                vis_frame[h-thumb_h:h, w-thumb_w:w] = mask_thumb
                
                # 加个边框
                cv2.rectangle(vis_frame, (w-thumb_w, h-thumb_h), (w, h), (255, 255, 255), 1)
                cv2.putText(vis_frame, "HSV Mask", (w-thumb_w+5, h-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("RK3588 HSV Detection", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(NUM_PROCESSES):
            try: input_queue.put(None)
            except: pass
        for p in workers:
            p.terminate()

if __name__ == "__main__":
    # Linux/ARM 下必须设置 spawn 启动方式
    mp.set_start_method('spawn', force=True)
    main()