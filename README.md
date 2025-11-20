# 🎯 基于视觉定位与云台控制的移动激光打击平台

## 1. 项目简介 (Introduction)

本项目旨在设计并制作一个能够自动识别、追踪并打击目标的移动激光平台。系统结合了嵌入式运动控制与边缘端计算机视觉技术，能够自主搜索环境中的特定目标（如白色圆形标靶、无人机或机器狗），并通过二自由度云台控制激光进行精准打击。

### 系统主要组成

**👁️ 视觉感知端（上位机）：**
- 运行在 RK3588 (香橙派 5) 或 PC 上
- 负责图像采集、YOLOv5/OpenCV 目标检测及坐标解算

**🦾 运动控制端（下位机）：**
- 运行在 ESP32 上
- 利用 SimpleFOC 驱动双路无刷电机，接收视觉信号并进行高频 PID 闭环控制

---

## 2. 系统架构 (System Architecture)

### 🛠 硬件清单

| 模块 | 设备/型号 | 说明 |
|------|-----------|------|
| 计算核心 | Orange Pi 5 (RK3588) / PC | 上位机，负责 AI 推理 |
| 主控芯片 | ESP32 (Lolin32 Lite) | 下位机，负责运动控制 |
| 电机驱动 | 灯哥开源 FOC V3.0 | SimpleFOC 驱动板 |
| 执行器 | 2804 / 3505 无刷云台电机 | 2x (Yaw轴 + Pitch轴) |
| 传感器 | AS5600 磁编码器 | I2C 接口，12-bit 精度 |
| 视觉传感器 | USB 免驱摄像头 | 支持 MJPG 格式 |
| 其他 | 激光模组、3D 打印件 | 云台结构与打击执行 |

### 💻 软件栈

- **下位机:** PlatformIO, C++, SimpleFOC Library
- **上位机:** Python 3.8+, OpenCV, RKNN-Toolkit2 (NPU 加速), PySerial

---

## 3. 目录结构说明 (Directory Structure)

```
laser-main/
├── FOC_Control_V2/                    # [下位机] ESP32 云台控制固件
│   ├── platformio.ini                 # PlatformIO 项目配置文件
│   ├── src/
│   │   └── main.cpp                   # 核心源码：PID算法、FOC配置、串口解析
│   └── lib/                           # 项目依赖库
├── gongchuang/                        # [上位机] 视觉识别与通信代码
│   ├── circle_detect_send_threads.py # 方案A：多线程OpenCV几何图形识别（基础任务）
│   ├── yolo_detect_rknn.py           # 方案B：RK3588 NPU加速 YOLO检测（发挥任务）
│   ├── best.rknn                      # 预编译的 YOLO 模型文件
│   ├── serial_connect.py              # 串口通信测试工具
│   ├── start_rknn.sh                  # 启动脚本
│   └── *.py                           # 其他辅助脚本
└── README.md                          # 项目说明文档
```

---

## 4. 功能模块详解

### 4.1 下位机固件 (FOC_Control_V2)

核心代码位于 `FOC_Control_V2/src/main.cpp`，主要特性：

- **VisualPID 算法:** 专为视觉伺服优化的 PID 类，修复了积分限幅 Bug，支持平滑追踪
- **安全传感器封装:** 自定义 `SafeMagneticSensorI2C` 类，过滤 I2C 读取错误和数据跳变
- **双轴控制:**
  - **M1 (Yaw/底座):** 负责水平旋转
  - **M2 (Pitch/俯仰):** 负责垂直运动，针对重力影响调整了 P 参数
- **指令解析:** 实时解析串口数据流，更新目标角度

### 4.2 上位机视觉 (gongchuang)

#### 🔵 方案 A: 几何图形检测 (基础部分)

- **入口文件:** `circle_detect_send_threads.py`
- **原理:** 使用 OpenCV 进行灰度化、二值化和轮廓分析。通过 Circularity (圆形度) 筛选白色圆形标靶
- **特性:**
  - **多线程架构:** 采集线程 + 发送线程分离，保证高帧率
  - **自动归位:** 若 5 秒 (`IDLE_TIME`) 未检测到目标，自动发送指令让云台回中

#### 🔴 方案 B: 深度学习检测 (发挥部分)

- **入口文件:** `yolo_detect_rknn.py`
- **原理:** 调用 RK3588 的 NPU 运行 YOLO 模型 (`best.rknn`)
- **特性:**
  - **多核并行:** 启动 3 个推理进程充分利用 NPU 算力
  - **大目标过滤:** 设定 `MAX_AREA_RATIO` (如 0.6)，当识别框过大（如贴脸误检）时自动忽略，防止云台乱动

---

## 5. 快速开始 (Quick Start)

### 5.1 下位机烧录 (ESP32)

1. 确保已安装 VS Code 及 PlatformIO 插件
2. 打开 `FOC_Control_V2` 文件夹
3. 连接 ESP32 开发板
4. 点击 PlatformIO 栏的 **Build (√)** 和 **Upload (→)**
5. 烧录完成后打开串口监视器 (115200)，复位开发板

> ✅ **成功标志：** 输出 `Home Set: M1=..., M2=...`

### 5.2 上位机运行 (RK3588/PC)

#### 环境依赖

```bash
pip install opencv-python pyserial numpy
# 如果是 RK3588，还需安装 rknn-toolkit-lite2 (见 gongchuang 目录下的 whl 文件)
```

#### 运行基础追踪 (方案 A)

```bash
cd gongchuang
# 确认 USB 转串口设备路径，例如 /dev/ttyUSB0
python3 circle_detect_send_threads.py
```

#### 运行 YOLO 识别 (方案 B - RK3588 Only)

```bash
cd gongchuang
# 运行 NPU 加速检测
python3 yolo_detect_rknn.py
```

---

## 6. 通信协议 (Communication Protocol)

上位机与 ESP32 通过 USB 串口通信，波特率 **115200**。

| 指令功能 | 格式 | 示例 | 说明 |
|----------|------|------|------|
| 视觉追踪 | `$target,X/Y&` | `$target,320/240&` | 发送目标在图像中的坐标中心 (X, Y)。下位机根据此坐标计算 PID 误差。 |
| 绝对角度 | `$Angle_x,Value#` | `$Angle_0,0.0#` `$Angle_1,1.5#` | 强制电机转动到指定弧度。0=Yaw轴, 1=Pitch轴。常用于丢失目标后的自动归位。 |
| PID 调试 | `$VPID_X,P/I/D*` | `$VPID_X,0.0004/0.0001/0*` | 在线修改 X 轴的 PID 参数，无需重新烧录。 |

---

## 7. 常见问题 (FAQ)

### Q1: 电机上电后狂转或震动？

- 检查 AS5600 磁铁是否安装居中且距离合适（1-2mm）
- 检查 `main.cpp` 中电机极对数 (`MOTOR0_PP`) 配置是否正确
- 建议顺序：先给 12V 驱动板上电，再给 ESP32 插 USB

### Q2: 视觉识别帧率低？

- 若使用 USB 摄像头，请确保连接在 **USB 3.0** 接口
- 在代码中尝试降低分辨率或开启 MJPG 模式

### Q3: 云台追踪方向相反？

- **不要**直接修改电机线序
- 请在 `main.cpp` 的 `updateVisualServoing` 函数中，修改 `target_angle` 的加减符号（例如由 `+=` 改为 `-=`）

---

## 8. 许可证 (License)

本项目代码仅供学习与比赛交流使用。遵循 **MIT License**。
