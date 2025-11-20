import cv2
import numpy as np
import time
from collections import defaultdict

class CircleDetector:
    def __init__(self, mode=1):
        """
        初始化圆形检测器
        mode: 1 - 每5秒统一返回一组坐标, 2 - 持续返回模式
        """
        self.mode = mode
        self.cap = cv2.VideoCapture(4)
        
        # 检测参数
        self.white_lower = np.array([200, 200, 200])  # 白色下界(BGR)
        self.white_upper = np.array([255, 255, 255])  # 白色上界(BGR)
        self.min_radius = 5
        self.max_radius = 200
        self.merge_threshold = 30  # 圆心距离阈值，用于去重
        self.min_circularity = 0.8  # 最小圆形度
        
        # 性能监控
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 模式1专用：定时返回
        self.last_report_time = time.time()
        self.report_interval = 5.0  # 5秒间隔
        self.next_report_time = self.last_report_time + self.report_interval
        
    def is_duplicate(self, center, detected_list, threshold):
        """检查圆心是否重复"""
        for existing_center in detected_list:
            distance = np.sqrt((center[0] - existing_center[0])**2 + 
                             (center[1] - existing_center[1])**2)
            if distance < threshold:
                return True
        return False
    
    def detect_white_circles(self, frame):
        """
        检测白色圆形
        返回: 圆心坐标列表, RGB处理图像, 灰度处理图像
        """
        # RGB处理
        mask_rgb = cv2.inRange(frame, self.white_lower, self.white_upper)
        
        # 灰度处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 结合两种mask提高准确度
        combined_mask = cv2.bitwise_and(mask_rgb, mask_gray)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 检测轮廓
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        frame_rgb_vis = frame.copy()
        frame_gray_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤太小的区域
                continue
            
            # 计算圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # 只保留圆形度高的轮廓
            if circularity > self.min_circularity:
                # 最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # 半径范围检查
                if self.min_radius < radius < self.max_radius:
                    # 去重检查
                    if not self.is_duplicate(center, circles, self.merge_threshold):
                        circles.append(center)
                        
                        # 在RGB图上绘制
                        cv2.circle(frame_rgb_vis, center, radius, (0, 255, 0), 2)
                        cv2.circle(frame_rgb_vis, center, 3, (0, 0, 255), -1)
                        
                        # 在灰度图上绘制
                        cv2.circle(frame_gray_vis, center, radius, (0, 255, 0), 2)
                        cv2.circle(frame_gray_vis, center, 3, (0, 0, 255), -1)
        
        return circles, frame_rgb_vis, frame_gray_vis
    
    def add_info_overlay(self, frame, text_lines, position='top'):
        """在图像上添加信息文本"""
        y_offset = 30 if position == 'top' else frame.shape[0] - 30
        for i, text in enumerate(text_lines):
            y_pos = y_offset + (i * 30) if position == 'top' else y_offset - (len(text_lines) - 1 - i) * 30
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def run(self):
        """主运行循环"""
        print(f"启动模式 {self.mode}")
        if self.mode == 1:
            print(f"模式1: 每{self.report_interval}秒统一返回一组所有检测到的圆形坐标")
        else:
            print("模式2: 持续返回每帧检测到的圆形坐标")
        print("按 'q' 退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            
            # 开始计时
            detection_start = time.time()
            
            # 检测圆形
            circles, rgb_vis, gray_vis = self.detect_white_circles(frame)
            
            # 计算检测时长
            detection_time = (time.time() - detection_start) * 1000  # 毫秒
            
            # 更新FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            current_time = time.time()
            
            # 处理不同模式
            if self.mode == 1:
                # 模式1: 每5秒返回一组
                time_until_report = self.next_report_time - current_time
                
                if current_time >= self.next_report_time:
                    if circles:
                        circle_coords = [(c[0], c[1]) for c in circles]
                        print(f"[模式1] 检测到 {len(circles)} 个白色圆形")
                        print(f"圆心坐标: {circle_coords}")
                    else:
                        print(f"[模式1] 未检测到白色圆形")
                    
                    # 更新下次报告时间
                    self.next_report_time = current_time + self.report_interval
                
                # 添加倒计时信息
                info_rgb = [
                    f"Mode: 1 (Every {self.report_interval}s)",
                    f"Next Report: {time_until_report:.1f}s",
                    f"FPS: {self.fps:.1f}",
                    f"Detection: {detection_time:.1f}ms",
                    f"Circles: {len(circles)}",
                ]
            else:
                # 模式2: 持续返回
                if circles:
                    circle_coords = [(c[0], c[1]) for c in circles]
                    print(f"[模式2] 检测到 {len(circles)} 个圆形: {circle_coords}")
                
                info_rgb = [
                    f"Mode: 2 (Continuous)",
                    f"FPS: {self.fps:.1f}",
                    f"Detection: {detection_time:.1f}ms",
                    f"Circles: {len(circles)}",
                ]
            
            info_gray = ["Grayscale View"]
            
            self.add_info_overlay(rgb_vis, info_rgb, 'top')
            self.add_info_overlay(gray_vis, info_gray, 'top')
            
            # 添加视图标签
            cv2.putText(rgb_vis, "RGB Detection", (10, rgb_vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(gray_vis, "Grayscale Detection", (10, gray_vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # 并排显示
            combined = np.hstack((rgb_vis, gray_vis))
            
            # 显示窗口
            cv2.imshow('Circle Detection - RGB | Grayscale', combined)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # 清理
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("\n请选择检测模式:")
    print("1. 定时返回模式 - 每5秒统一返回一组所有圆形坐标")
    print("2. 持续返回模式 - 持续返回每帧检测到的圆形坐标")
    while True:
        try:
            mode = int(input("\n请输入模式 (1 或 2): "))
            if mode in [1, 2]:
                break
            else:
                print("请输入 1 或 2")
        except ValueError:
            print("输入无效，请输入数字 1 或 2")
    detector = CircleDetector(mode=mode)
    detector.run()

if __name__ == "__main__":
    main()