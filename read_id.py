import serial
import time

# --- 配置区域 ---
SERIAL_PORT = 'COM8'      # 请修改为你的实际串口号 (Windows: COMx, Linux: /dev/ttyUSBx)
BAUD_RATE = 115200       # 根据文档，通信速率通常为 1Mbps [cite: 4, 10]
TIMEOUT = 0.5          # 每个ID的等待超时时间(秒)，适当调整

def calculate_checksum(id_val, length, instruction, parameters=[]):
    """
    根据文档计算校验和: ~(ID + Length + Instruction + Parameters)
    参考 [cite: 60, 61]
    """
    checksum_sum = id_val + length + instruction + sum(parameters)
    # 取低8位并取反
    checksum = (~checksum_sum) & 0xFF
    return checksum

def send_ping(ser, target_id):
    """
    发送 PING 指令查询单个 ID 的状态
    PING 指令包格式: [字头H, 字头L, ID, 长度, 指令, 校验和]
    参考 [cite: 80, 81]
    """
    header_h = 0x12
    header_l = 0x4C
    length = 0x02         # PING指令长度固定为 2
    instruction = 0x01    # PING指令码
    
    checksum = calculate_checksum(target_id, length, instruction)
    
    # 构造字节串
    packet = bytes([header_h, header_l, target_id, length, instruction, checksum])
    
    # 清空输入缓冲区，防止读取到旧数据
    ser.reset_input_buffer()
    ser.write(packet)

def read_response(ser):
    """
    尝试读取舵机的应答包
    应答包格式: [字头H, 字头L, ID, 长度, 状态(Error), 参数..., 校验和]
    参考 [cite: 63, 64]
    """
    # 读取包头 0x12 0x4C
    start_time = time.time()
    header = []
    
    while True:
        if time.time() - start_time > TIMEOUT:
            return None # 超时未收到数据
        
        if ser.in_waiting > 0:
            byte = ser.read(1)[0]
            if len(header) == 0 and byte == 0x12:
                header.append(byte)
            elif len(header) == 1:
                if byte == 0x4C:
                    header.append(byte)
                    break
                else:
                    header = [] # 如果第二个字节不是0x4C，重置
    
    if len(header) != 2:
        return None

    # 读取 ID, Length
    id_len = ser.read(2)
    if len(id_len) < 2: 
        return None
        
    motor_id = id_len[0]
    length = id_len[1]
    
    # 读取剩余数据: (Length - 2)个参数 + 1个Error + 1个Checksum = Length 字节? 
    # 等等，文档说 Length = 参数个数 + 2 [cite: 56]。
    # 应答包 Length 定义：Error + Parameters + Checksum??
    # 让我们看例1应答[cite: 84]: 12 4C 01 02 00 FC
    # ID=01, Len=02. 后面跟了 00 (Error) 和 FC (Checksum). 
    # 所以应答包剩余字节数 = Length.
    
    remaining_bytes = ser.read(length)
    if len(remaining_bytes) < length:
        return None
        
    error_byte = remaining_bytes[0]
    checksum_received = remaining_bytes[-1]
    
    # 这里可以加校验和验证逻辑，但为了扫描速度，只要格式对通常就是读到了
    return motor_id

def main():
    try:
        # 打开串口
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"成功打开串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")
        print("开始扫描总线上的电机 ID (范围 0-253)...")
        print("-" * 30)

        found_motors = []

        # 遍历所有可能的 ID 
        for check_id in range(254): 
            # 发送 PING
            send_ping(ser, check_id)
            
            # 等待并解析回复
            found_id = read_response(ser)
            
            if found_id is not None:
                # 再次确认 ID 匹配
                if found_id == check_id:
                    print(f"发现电机! ID: {found_id} (0x{found_id:02X})")
                    found_motors.append(found_id)
                else:
                    print(f"收到异常回复: 请求ID {check_id} 但收到 {found_id}")
            
            # 简单的延时防止总线拥堵（虽然是半双工问答式，但稍微停顿更稳）
            # time.sleep(0.005) 

        print("-" * 30)
        print(f"扫描结束。共发现 {len(found_motors)} 个电机。")
        print(f"ID 列表: {found_motors}")

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()