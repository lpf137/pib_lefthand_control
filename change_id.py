import serial
import time

# --- 配置区域 ---
SERIAL_PORT = 'COM4'      # 请修改为你的实际串口号 (Windows: COMx, Linux: /dev/ttyUSBx)
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

def send_packet(ser, target_id, instruction, parameters=[]):
    header_h = 0x12
    header_l = 0x4C
    # Length = Parameter Count + 2
    length = len(parameters) + 2
    
    checksum = calculate_checksum(target_id, length, instruction, parameters)
    
    # 构造字节串
    packet = bytes([header_h, header_l, target_id, length, instruction] + parameters + [checksum])
    
    # 清空输入缓冲区
    ser.reset_input_buffer()
    ser.write(packet)

def send_ping(ser, target_id):
    """
    发送 PING 指令
    """
    # PING指令码 0x01
    send_packet(ser, target_id, 0x01, [])

def send_write_byte(ser, target_id, address, value):
    """
    发送 WRITE 指令 (0x03) 写一个字节
    """
    # 参数: [地址, 值]
    send_packet(ser, target_id, 0x03, [address, value])

def send_read_byte(ser, target_id, address):
    """
    发送 READ 指令 (0x02) 读一个字节
    """
    # 参数: [地址, 读取长度]
    send_packet(ser, target_id, 0x02, [address, 0x01])

def read_response(ser):
    """
    尝试读取舵机的应答包
    返回: (motor_id, error_byte, parameters)
    """
    # 读取包头 0x12 0x4C
    start_time = time.time()
    header = []
    
    while True:
        if time.time() - start_time > TIMEOUT:
            return None, None, None # 超时
        
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
        return None, None, None

    # 读取 ID, Length
    id_len = ser.read(2)
    if len(id_len) < 2: 
        return None, None, None
        
    motor_id = id_len[0]
    length = id_len[1]
    
    # 读取剩余数据: Length 字节 (Error + Params + Checksum)
    remaining_bytes = ser.read(length)
    if len(remaining_bytes) < length:
        return None, None, None
        
    error_byte = remaining_bytes[0]
    params = remaining_bytes[1:-1] # 中间的是参数
    checksum_received = remaining_bytes[-1]
    
    return motor_id, error_byte, params

def main():
    try:
        # 打开串口
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"成功打开串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")
        
        # --- 修改 ID 逻辑 ---
        old_id = 16
        new_id = 10
        print(f"准备将电机 ID 从 {old_id} 修改为 {new_id}...")
        
        # 1. 尝试猜测 ID 所在的寄存器地址 (通常是 3 或 5)
        id_addr_candidates = [5, 3]
        target_addr = None
        
        for addr in id_addr_candidates:
            print(f"尝试读取地址 {addr}...")
            send_read_byte(ser, old_id, addr)
            rid, err, params = read_response(ser)
            
            if rid == old_id and params and len(params) > 0:
                val = params[0]
                print(f"地址 {addr} 的值为: {val}")
                if val == old_id:
                    target_addr = addr
                    print(f"--> 确认 ID 寄存器地址为: {addr}")
                    break
        
        if target_addr is None:
            print("错误: 无法确认 ID 寄存器地址，或无法连接到电机。")
            return

        # 2. 写入新 ID
        print(f"正在写入新 ID {new_id} 到地址 {target_addr}...")
        send_write_byte(ser, old_id, target_addr, new_id)
        time.sleep(0.1) # 等待写入生效
        
        # 3. 验证
        print(f"正在验证新 ID {new_id}...")
        send_ping(ser, new_id)
        rid, err, params = read_response(ser)
        
        if rid == new_id:
            print(f"成功! 发现电机 ID {new_id}。修改完成。")
        else:
            print("验证失败，未收到新 ID 的响应。")

        # 恢复扫描功能 (可选)
        # print("-" * 30)
        # print("重新扫描总线...")
        # ... (原有扫描代码)

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()