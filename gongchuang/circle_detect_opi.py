import cv2
import numpy as np
import time  # 1. 引入 time 库

# ================= 全局参数设置 =================
# 在这里直接修改数值即可调整程序行为
THRESH_MIN = 90       # 二值化阈值
CIRCULARITY_PARAM = 85 # 圆形度阈值 (85 代表 0.85)
AREA_MIN = 100        # 最小面积阈值
# ==============================================

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print(f"当前参数 -> 阈值: {THRESH_MIN}, 圆形度: {CIRCULARITY_PARAM/100.0}, 最小面积: {AREA_MIN}")
    print("按 'q' 键退出程序")

    # 2. 初始化帧率计算用的时间变量
    pTime = 0 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. 计算 FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        # 1.直接使用全局变量赋值
        thresh_val = THRESH_MIN
        circ_val = CIRCULARITY_PARAM / 100.0 
        area_val = AREA_MIN

        # 2. 图像处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 3. 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_val:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)

            # 如果符合圆形度，画绿框；不符合画红框
            if circularity > circ_val:
                color = (0, 255, 0) # 绿色 - 成功
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.circle(vis_frame, (int(x), int(y)), int(radius), color, 2)
                cv2.putText(vis_frame, f"{circularity:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # 绘制不符合圆形度的轮廓
                cv2.drawContours(vis_frame, [contour], -1, (0, 0, 255), 1)

        # 缩小显示方便查看
        h, w = frame.shape[:2]
        small_frame = cv2.resize(vis_frame, (w//2, h//2))
        small_mask = cv2.resize(mask, (w//2, h//2))
        small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR) # 转为3通道以便拼接

        # 拼接显示
        combined = np.hstack((small_frame, small_mask))
        
        # 在界面上显示参数
        info_text = f"Thresh: {thresh_val} | Circ: {circ_val} | Area: {area_val}"
        cv2.putText(combined, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. 显示 FPS (绿色字体)
        cv2.putText(combined, f"FPS: {int(fps)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Result Panel", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()