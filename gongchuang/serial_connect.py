import serial_connect
import time


# ================= 公共工具函数 =================

def build_frame(data_type, values):
    """
    根据协议拼接一帧数据字符串
    例如: build_frame("angles", [1.02, 1.03]) -> "$angles,1.02/1.03&"
    """
    data_str = "/".join(str(v) for v in values)  # 数据内容之间用 /
    frame = f"${data_type},{data_str}&"          # 类型和内容之间用 ,
    return frame


def parse_frame(frame_body):
    """
    解析一帧数据（不包含起始 '$' 和结束 '&'）
    frame_body 示例: "angles,1.02/1.03"
    返回: (data_type, data_list)
    """
    try:
        data_type, data_str = frame_body.split(",", 1)
    except ValueError:
        # 没有逗号，格式错误
        return None, None

    data_list = data_str.split("/")  # 内容之间用 /
    return data_type, data_list


# ================ 串口发送函数 =================

def send_loop(port="COM3", baudrate=115200):
    """
    持续发送示例数据的发送循环
    """
    ser = serial_connect.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial_connect.EIGHTBITS,
        parity=serial_connect.PARITY_NONE,
        stopbits=serial_connect.STOPBITS_ONE,
        timeout=0.5,
    )

    print("发送端串口已打开:", ser.port)

    try:
        while True:
            # 构造一帧示例数据：类型为 angles，两个浮点数
            values = [1.02, 1.03]
            frame = build_frame("angles", values)
            print("发送数据:", frame)

            ser.write(frame.encode("ascii"))  # 串口发送需要 bytes
            time.sleep(1)  # 间隔 1 秒发送一次

    except KeyboardInterrupt:
        print("发送停止（Ctrl+C）")

    finally:
        ser.close()
        print("发送端串口已关闭")


# ================ 串口接收函数 =================

def receive_loop(port="COM4", baudrate=115200):
    """
    持续接收并解析数据的接收循环
    """
    ser = serial_connect.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial_connect.EIGHTBITS,
        parity=serial_connect.PARITY_NONE,
        stopbits=serial_connect.STOPBITS_ONE,
        timeout=0.1,
    )

    print("接收端串口已打开:", ser.port)

    buffer = ""  # 用来累积接收的字符串数据

    try:
        while True:
            if ser.in_waiting:  # 缓冲区有数据
                raw = ser.read(ser.in_waiting)
                text = raw.decode("ascii", errors="ignore")
                buffer += text

                # 循环提取所有完整帧
                while True:
                    start = buffer.find("$")
                    end = buffer.find("&", start + 1)

                    # 没有找到完整的一帧
                    if start == -1 or end == -1:
                        # 如果 '$' 前还有垃圾数据，把它丢掉
                        if start > 0:
                            buffer = buffer[start:]
                        break

                    # 取出一帧（不包含 $ 和 &）
                    frame_body = buffer[start + 1:end]
                    # 剩余数据留在 buffer 里
                    buffer = buffer[end + 1:]

                    # 解析这一帧
                    data_type, data_list = parse_frame(frame_body)
                    if data_type is None:
                        print("收到格式错误的帧:", frame_body)
                    else:
                        print(f"类型:{data_type},内容:{data_list}")
    except KeyboardInterrupt:
        print("接收停止（Ctrl+C）")

    finally:
        ser.close()
        print("接收端串口已关闭")

def main():
    1

if __name__=="__main__":
    1