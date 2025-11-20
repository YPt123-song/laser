import cv2
import time
import numpy as np
import multiprocessing as mp
from queue import Empty
from rknn.api import RKNN

# =================配置参数=================
RKNN_MODEL_PATH = 'best.rknn'  # 模型路径
IMG_SIZE = 640                 # 模型输入尺寸
CONF_THRES = 0.5               # 置信度阈值 (过滤低分物体)
IOU_THRES = 0.45               # NMS 阈值 (重叠度超过0.45则视为重复框)
NPU_CORES = 3                  # 启动的推理进程数

# [新增配置] 最大面积占比阈值 (0.0 ~ 1.0)
# 例如 0.6 表示：如果目标框面积超过屏幕总面积的 60%，则将其忽略
MAX_AREA_RATIO = 0.6          
# =========================================

# -----------------------------------------
# 独立函数：Letterbox (预处理)
# -----------------------------------------
def letterbox(img, new_size=IMG_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    dw = (new_size - nw) // 2
    dh = (new_size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw, :] = resized
    
    return canvas, scale, dw, dh

# -----------------------------------------
# 独立函数：Postprocess (后处理 + NMS 去重 + 尺寸过滤)
# -----------------------------------------
def postprocess(outputs, orig_shape, scale, dw, dh, conf_thres, iou_thres, max_area_ratio):
    """
    解析模型输出并进行 NMS 去重，同时过滤过大目标
    """
    h0, w0 = orig_shape[:2]
    frame_area = h0 * w0  # 原图总面积用于计算占比

    out = outputs[0]
    out = np.array(out)

    # 统一维度
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    
    if out.shape[0] != 5:
        return [], []

    # 1. 提取数据
    cx = out[0, :]
    cy = out[1, :]
    w = out[2, :]
    h = out[3, :]
    conf = out[4, :]

    # 2. 初步置信度过滤
    mask = conf >= conf_thres
    cx, cy, w, h, conf = cx[mask], cy[mask], w[mask], h[mask], conf[mask]

    if len(conf) == 0:
        return [], []

    # 3. 坐标还原 (还原到 letterbox 之前的原图坐标)
    x1_p = cx - w / 2
    y1_p = cy - h / 2
    x2_p = cx + w / 2
    y2_p = cy + h / 2

    # 去掉 padding 并缩放
    x1 = (x1_p - dw) / scale
    y1 = (y1_p - dh) / scale
    x2 = (x2_p - dw) / scale
    y2 = (y2_p - dh) / scale
    
    # 映射回来的 w, h
    w_orig = x2 - x1
    h_orig = y2 - y1

    # 限制在图像范围内
    x1 = np.clip(x1, 0, w0 - 1)
    y1 = np.clip(y1, 0, h0 - 1)
    
    # 4. 准备 NMS 数据
    boxes_xywh = []
    scores = []

    for i in range(len(conf)):
        # cv2 NMS 需要整数坐标 [x, y, w, h]
        x_int = int(x1[i])
        y_int = int(y1[i])
        w_int = int(w_orig[i])
        h_int = int(h_orig[i])

        # ---------------- [新增逻辑] 大目标过滤 ----------------
        # 计算当前框的面积
        box_area = w_int * h_int
        
        # 如果 (目标面积 / 屏幕总面积) > 阈值，则认为是异常大框，跳过
        if (box_area / frame_area) > max_area_ratio:
            continue
        # -----------------------------------------------------

        boxes_xywh.append([x_int, y_int, w_int, h_int])
        scores.append(float(conf[i]))

    # 5. 执行 NMS (核心去重步骤)
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)

    final_boxes = []
    final_centers = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_xywh[i]
            score = scores[i]

            # 转回 x1, y1, x2, y2 方便画图
            x1_draw = x
            y1_draw = y
            x2_draw = x + w
            y2_draw = y + h
            
            final_boxes.append([x1_draw, y1_draw, x2_draw, y2_draw])
            
            # 计算中心点
            cx_c = int(x + w / 2)
            cy_c = int(y + h / 2)
            final_centers.append((cx_c, cy_c, score))

    return final_boxes, final_centers

# -----------------------------------------
# 核心类：推理工作进程
# -----------------------------------------
class InferenceWorker(mp.Process):
    def __init__(self, input_queue, output_queue, model_path):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_path = model_path
        self.daemon = True 

    def run(self):
        rknn = RKNN(verbose=False)
        if rknn.load_rknn(self.model_path) != 0:
            print("Load RKNN failed!")
            return
        if rknn.init_runtime(target='rk3588') != 0:
            print("Init runtime failed!")
            return

        while True:
            try:
                task = self.input_queue.get() 
                if task is None: break
                
                frame_id, frame = task

                # 1. 预处理
                img_input, scale, dw, dh = letterbox(frame, IMG_SIZE)
                img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                img_input = np.expand_dims(img_input, 0)

                # 2. 推理
                outputs = rknn.inference(inputs=[img_input])

                # 3. 后处理 (包含 NMS 和 面积过滤)
                # 传入 MAX_AREA_RATIO
                boxes, centers = postprocess(
                    outputs, 
                    frame.shape, 
                    scale, 
                    dw, 
                    dh, 
                    CONF_THRES, 
                    IOU_THRES,
                    MAX_AREA_RATIO # <--- 传入全局配置
                )

                # 4. 返回结果
                self.output_queue.put((frame_id, boxes, centers))

            except Exception as e:
                print(f"Worker Error: {e}")

        rknn.release()

# -----------------------------------------
# 主程序
# -----------------------------------------
def main():
    # 通信队列
    input_queue = mp.Queue(maxsize=NPU_CORES + 2)
    output_queue = mp.Queue(maxsize=NPU_CORES + 2)

    # 启动进程
    workers = []
    print(f"启动 {NPU_CORES} 个 NPU 进程，已启用 NMS 及大目标过滤 (阈值:{MAX_AREA_RATIO})...")
    for i in range(NPU_CORES):
        p = InferenceWorker(input_queue, output_queue, RKNN_MODEL_PATH)
        p.start()
        workers.append(p)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    current_boxes = []
    current_centers = []
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_id += 1

            # 发送
            if not input_queue.full():
                input_queue.put((frame_id, frame))

            # 接收
            try:
                while not output_queue.empty():
                    _, boxes, centers = output_queue.get_nowait()
                    current_boxes = boxes
                    current_centers = centers
            except Empty:
                pass

            # 绘制
            vis_frame = frame.copy()
            for (x1, y1, x2, y2), (_, _, conf) in zip(current_boxes, current_centers):
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_frame, f'{conf:.2f}', (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 统计
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("RK3588 NMS Detection", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(NPU_CORES):
            try: input_queue.put(None)
            except: pass
        for p in workers:
            p.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()