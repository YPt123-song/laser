import cv2
import time
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, mode=1, conf_thres=0.5, model_path="yolov8n.pt"):
        """
        YOLO 目标检测器

        :param mode: 1 - 只返回一次坐标；2 - 持续返回坐标
        :param conf_thres: 置信度阈值 (0~1)
        :param model_path: YOLO 模型路径
        """
        self.mode = mode
        self.conf_thres = conf_thres
        self.model = YOLO(model_path)

        # 打开摄像头
        self.cap = cv2.VideoCapture(4)

        # 性能监控
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        # 模式1专用：确保只返回一次坐标
        self.first_report_done = False

    def add_info_overlay(self, frame, text_lines):
        """在图像左上角叠加文字信息"""
        y = 30
        for text in text_lines:
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),  # 黄绿色字体
                2,
            )
            y += 30

    def run(self):
        print(f"\n当前模式: {self.mode}")
        print(f"置信度阈值: {self.conf_thres:.2f}")
        print("按 'q' 退出程序\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头")
                break

            # 记录检测开始时间
            det_start = time.time()

            # YOLO 推理，使用官方可视化
            # results 为一个列表，每次只喂一帧，所以取 [0]
            results = self.model(frame, conf=self.conf_thres, verbose=False)[0]

            # 检测时长（毫秒）
            det_time_ms = (time.time() - det_start) * 1000

            # 从结果中取出目标中心坐标
            centers = []
            boxes = results.boxes  # Boxes 对象
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # [N, 4] -> x1,y1,x2,y2
                confs = boxes.conf.cpu().numpy()  # [N]
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    centers.append((cx, cy, float(conf)))

            # 使用 YOLO 官方画图接口，可视化检测结果
            vis_frame = results.plot()  # 已画好 bbox、类别和置信度

            # 计算 FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

            # 模式逻辑处理
            if self.mode == 1:
                # 模式1：只在第一次检测到目标时打印一次坐标
                if (not self.first_report_done) and centers:
                    simple_centers = [(c[0], c[1]) for c in centers]
                    print(f"[模式1] 检测到 {len(simple_centers)} 个目标")
                    print(f"[模式1] 目标中心像素坐标(只打印一次): {simple_centers}")
                    self.first_report_done = True

                mode_text = "Mode: 1 (只返回一次坐标)"
            else:
                # 模式2：每帧持续打印坐标
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

            # 显示窗口
            cv2.imshow("YOLO Detection", vis_frame)

            # 按 'q' 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # 资源释放
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    print("\n===== YOLO 目标检测 =====")
    print("请选择检测模式:")
    print("1. 模式1：只在检测到目标后返回一次所有目标中心像素坐标")
    print("2. 模式2：持续返回每帧检测到的目标中心像素坐标")

    # 选择模式
    while True:
        try:
            mode = int(input("\n请输入模式 (1 或 2): "))
            if mode in (1, 2):
                break
            else:
                print("请输入 1 或 2")
        except ValueError:
            print("输入无效，请输入数字 1 或 2")

    # 选择置信度阈值
    conf_input = input("请输入置信度阈值 (0~1，直接回车默认 0.5): ").strip()
    if conf_input == "":
        conf_thres = 0.5
    else:
        try:
            conf_thres = float(conf_input)
        except ValueError:
            print("输入无效，使用默认置信度阈值 0.5")
            conf_thres = 0.5

    # 限制在 0.01~0.99 之间，避免极端值
    conf_thres = max(0.01, min(conf_thres, 0.99))

    detector = YOLODetector(mode=mode, conf_thres=conf_thres, model_path="runs/detect/train3/weights/best.pt")
    detector.run()


if __name__ == "__main__":
    main()
