import cv2
import time
import numpy as np
from rknn.api import RKNN   # rknn-toolkit2

RKNN_MODEL_PATH = 'best.rknn'  # 或你的实际 rknn 模型
IMG_SIZE = 640                 # 必须和导出 rknn 时的 imgsz 一致


class RKNNYOLODetector:
    def __init__(self, mode=2, conf_thres=0.3, rknn_path=RKNN_MODEL_PATH):
        """
        基于 RKNN 的 YOLO 目标检测器（在 rk3588 上本地推理）

        :param mode: 1 - 只返回一次坐标；2 - 持续返回坐标
        :param conf_thres: 置信度阈值 (0~1)，先用 0.3 稍低一点
        :param rknn_path: rknn 模型路径
        """
        self.mode = mode
        self.conf_thres = conf_thres

        # -------- 1) 初始化 RKNN --------
        print(f'[RKNN] Loading RKNN model from {rknn_path}')
        self.rknn = RKNN(verbose=False)
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f'Load RKNN model failed! ret={ret}')

        print('[RKNN] Init runtime (target=rk3588)')
        ret = self.rknn.init_runtime(target='rk3588')
        if ret != 0:
            raise RuntimeError(f'Init runtime failed! ret={ret}')

        # -------- 2) 打开摄像头 --------
        self.cap = cv2.VideoCapture(4)  # 根据实际设备改 index
        if not self.cap.isOpened():
            raise RuntimeError('无法打开摄像头 /dev/video4 (index=4)')

        # 性能监控
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        # 模式1：只返回一次
        self.first_report_done = False

        # 调试计数
        self.debug_printed = False

    # ==============================
    # 在图像上叠加文字
    # ==============================
    def add_info_overlay(self, frame, text_lines):
        y = 30
        for text in text_lines:
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y += 30

    # ==============================
    # letterbox 到 IMG_SIZE×IMG_SIZE
    # ==============================
    def letterbox(self, img, new_size=IMG_SIZE, color=(114, 114, 114)):
        """
        等比例缩放 + padding 到正方形。
        返回：img_resize, scale, dw, dh
        """
        h, w = img.shape[:2]
        scale = min(new_size / h, new_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))

        # 缩放
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 填充到 new_size×new_size
        canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
        dw = (new_size - nw) // 2
        dh = (new_size - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw, :] = resized

        return canvas, scale, dw, dh

    # ==============================
    # 后处理：从 RKNN 输出解析成 bbox
    # ==============================
    def postprocess(self, outputs, orig_shape, scale, dw, dh):
        """
        将 rknn.inference 输出解析为 bbox+score+center。

        已知 raw output: (1, 5, 8400) float32
        这里将其解释为 (N, 5) = [x1, y1, x2, y2, conf_logit]，
        对 conf 先做 sigmoid，再按阈值筛选。
        """
        h0, w0 = orig_shape[:2]

        out = outputs[0]
        out = np.array(out)

        # 调试：只打印一次形状和数值范围
        if not self.debug_printed:
            print('[DEBUG] raw output shape:', out.shape,
                  'dtype:', out.dtype,
                  'min:', float(out.min()), 'max:', float(out.max()))
            self.debug_printed = True

        # 去掉 batch 维度: (1, 5, 8400) -> (5, 8400)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]   # (5, 8400)

        # 现在是 (5, N)，转为 (N, 5)
        if out.ndim == 2:
            if out.shape[0] == 5:
                out = out.T   # (5, N) -> (N, 5)
            elif out.shape[1] == 5:
                pass          # (N, 5) 已 OK
            else:
                print('[WARN] 2D output but none of axes == 5, shape:', out.shape)
                return [], []
        else:
            print('[WARN] Unexpected output ndim:', out.ndim, 'shape:', out.shape)
            return [], []

        if out.shape[1] != 5:
            print('[WARN] out final shape not (N,5):', out.shape)
            return [], []

        # 解释为 [x1, y1, x2, y2, conf_logit]
        x1 = out[:, 0]
        y1 = out[:, 1]
        x2 = out[:, 2]
        y2 = out[:, 3]
        conf = out[:, 4]

        # 转成 float
        x1 = x1.astype(np.float32)
        y1 = y1.astype(np.float32)
        x2 = x2.astype(np.float32)
        y2 = y2.astype(np.float32)
        conf = conf.astype(np.float32)

        # 对 conf 做 sigmoid，假设其为 logit
        conf = 1.0 / (1.0 + np.exp(-conf))

        # 再调试一下 conf 的分布
        # （只输出一次，避免刷屏）
        print('[DEBUG] conf after sigmoid: min = %.4f, max = %.4f' %
              (float(conf.min()), float(conf.max())))

        # 置信度筛选
        mask = conf >= self.conf_thres
        x1 = x1[mask]
        y1 = y1[mask]
        x2 = x2[mask]
        y2 = y2[mask]
        conf = conf[mask]

        if len(conf) == 0:
            return [], []

        # 去掉 padding，再映射回原图
        x1 = (x1 - dw) / scale
        y1 = (y1 - dh) / scale
        x2 = (x2 - dw) / scale
        y2 = (y2 - dh) / scale

        # 限制在图像尺寸内
        x1 = np.clip(x1, 0, w0 - 1)
        y1 = np.clip(y1, 0, h0 - 1)
        x2 = np.clip(x2, 0, w0 - 1)
        y2 = np.clip(y2, 0, h0 - 1)

        boxes = np.stack([x1, y1, x2, y2], axis=-1)  # (N, 4)

        # 简单“伪 NMS”：按置信度排序保留前 200 个
        idxs = np.argsort(-conf)
        if len(idxs) > 200:
            idxs = idxs[:200]
        boxes = boxes[idxs]
        conf = conf[idxs]

        centers = []
        for b, c in zip(boxes, conf):
            x1b, y1b, x2b, y2b = b
            cx_c = int((x1b + x2b) / 2)
            cy_c = int((y1b + y2b) / 2)
            centers.append((cx_c, cy_c, float(c)))

        return boxes, centers

    # ==============================
    # 在图像上画框
    # ==============================
    def draw_boxes(self, frame, boxes, centers):
        for (x1, y1, x2, y2), (_, _, conf) in zip(boxes, centers):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ==============================
    # 主循环
    # ==============================
    def run(self):
        print(f"\n当前模式: {self.mode}")
        print(f"置信度阈值: {self.conf_thres:.2f}")
        print("按 'q' 退出程序\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头")
                break

            orig_h, orig_w = frame.shape[:2]

            # 预处理：letterbox 到 640×640
            img_input, scale, dw, dh = self.letterbox(frame, IMG_SIZE)

            # BGR -> RGB
            img_input_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

            # 保持 NHWC： (H, W, C) -> (1, H, W, C)
            img_input_rgb = img_input_rgb.astype(np.uint8)
            img_input_rgb = np.expand_dims(img_input_rgb, 0)

            # 记录检测开始时间
            det_start = time.time()

            # RKNN 推理
            outputs = self.rknn.inference(inputs=[img_input_rgb])

            det_time_ms = (time.time() - det_start) * 1000

            # 后处理：解析 bbox + centers
            boxes, centers = self.postprocess(outputs, frame.shape, scale, dw, dh)

            # 画框
            vis_frame = frame.copy()
            self.draw_boxes(vis_frame, boxes, centers)

            # 计算 FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

            # 模式逻辑
            if self.mode == 1:
                if (not self.first_report_done) and centers:
                    simple_centers = [(c[0], c[1]) for c in centers]
                    print(f"[模式1] 检测到 {len(simple_centers)} 个目标")
                    print(f"[模式1] 目标中心像素坐标(只打印一次): {simple_centers}")
                    self.first_report_done = True

                mode_text = "Mode: 1 (只返回一次坐标)"
            else:
                if centers:
                    simple_centers = [(c[0], c[1]) for c in centers]
                    print(f"[模式2] 检测到 {len(simple_centers)} 个目标")
                    print(f"[模式2] 目标中心像素坐标: {simple_centers}")

                mode_text = "Mode: 2 (持续返回坐标)"

            # 叠加信息到画面上
            info_lines = [
                mode_text,
                f"FPS: {self.fps:.1f}",
                f"Detection: {det_time_ms:.1f} ms",
                f"Detections: {len(centers)}",
                f"Conf Thres: {self.conf_thres:.2f}",
            ]
            self.add_info_overlay(vis_frame, info_lines)

            cv2.imshow("RKNN YOLO Detection", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.rknn.release()


def main():
    print("\n===== RKNN YOLO 目标检测 =====")
    mode_in = input("请输入模式 (1=一次返回，2=持续): ").strip()
    mode = 1 if mode_in != '2' else 2

    conf_input = input("请输入置信度阈值 (默认0.3): ").strip()
    if conf_input == "":
        conf_thres = 0.3
    else:
        try:
            conf_thres = float(conf_input)
        except ValueError:
            print("输入无效，使用默认 0.3")
            conf_thres = 0.3
    conf_thres = max(0.01, min(conf_thres, 0.99))

    detector = RKNNYOLODetector(
        mode=mode,
        conf_thres=conf_thres,
        rknn_path=RKNN_MODEL_PATH,
    )
    detector.run()


if __name__ == "__main__":
    main()
