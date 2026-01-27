import serial
import time

# --- 配置区域 ---
SERIAL_PORT = 'COM8'      # 请修改为你的实际串口号
BAUD_RATE = 115200       # 通信速率
TIMEOUT = 0.5          # 超时时间

# 12-bit position range: 0-4095
# 协议头: 0x12 0x4C

def calculate_checksum(id_val, length, instruction, parameters=[]):
    """
    计算校验和: ~(ID + Length + Instruction + Parameters) & 0xFF
    """
    checksum_sum = id_val + length + instruction + sum(parameters)
    checksum = (~checksum_sum) & 0xFF
    return checksum

def send_packet(ser, target_id, instruction, parameters=[]):
    header_h = 0x12
    header_l = 0x4C
    # Length = Parameter Count + 2
    length = len(parameters) + 2
    
    checksum = calculate_checksum(target_id, length, instruction, parameters)
    
    packet = bytes([header_h, header_l, target_id, length, instruction] + parameters + [checksum])
    
    ser.reset_input_buffer()
    ser.write(packet)

def read_response(ser):
    start_time = time.time()
    header = []
    
    while True:
        if time.time() - start_time > TIMEOUT:
            return None, None, None 
        
        if ser.in_waiting > 0:
            byte = ser.read(1)[0]
            if len(header) == 0 and byte == 0x12:
                header.append(byte)
            elif len(header) == 1:
                if byte == 0x4C:
                    header.append(byte)
                    break
                else:
                    header = [] 
    
    if len(header) != 2:
        return None, None, None

    id_len = ser.read(2)
    if len(id_len) < 2: 
        return None, None, None
        
    motor_id = id_len[0]
    length = id_len[1]
    
    remaining_bytes = ser.read(length)
    if len(remaining_bytes) < length:
        return None, None, None
        
    error_byte = remaining_bytes[0]
    params = remaining_bytes[1:-1]
    checksum_received = remaining_bytes[-1]
    
    return motor_id, error_byte, params

def read_position(ser, motor_id):
    # 根据用户抓包，位置地址是 0x38 (56)，长度 2 字节
    addr = 0x38
    read_len = 0x02
    
    # 指令 0x02 是 READ
    # 参数: [Address, Bytes_to_Read]
    send_packet(ser, motor_id, 0x02, [addr, read_len])
    
    rid, err, params = read_response(ser)
    
    if rid == motor_id and params and len(params) == read_len:
        # 根据抓包数据 07 F8 -> 2040，判定为大端序 (High Byte First)
        position = (params[0] << 8) + params[1]
        return position
    else:
        # print(f"Debugging: READ Fail. RID={rid}, Err={err}, Params={params}")
        return None

def read_min_angle_limit(ser, motor_id):
    # 最小角度限制地址 0x09 (9)，长度 2 字节
    addr = 0x09
    read_len = 0x02
    
    send_packet(ser, motor_id, 0x02, [addr, read_len])
    rid, err, params = read_response(ser)
    
    if rid == motor_id and params and len(params) == read_len:
        position = (params[0] << 8) + params[1]
        return position
    else:
        return None

def read_max_angle_limit(ser, motor_id):
    # 最大角度限制地址 0x0B (11)，长度 2 字节
    addr = 0x0B
    read_len = 0x02
    
    send_packet(ser, motor_id, 0x02, [addr, read_len])
    rid, err, params = read_response(ser)
    
    if rid == motor_id and params and len(params) == read_len:
        position = (params[0] << 8) + params[1]
        return position
    else:
        return None

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"成功打开串口 {SERIAL_PORT}")
        
        target_id = 0x12
        print(f"正在读取电机 ID {target_id} 的当前位置...")
        
        pos = read_position(ser, target_id)
        
        if pos is not None:
            angle = (pos / 4095.0) * 200.0
            print(f"------------------------------")
            print(f"电机 ID: {target_id}")
            print(f"当前位置: {pos} (0x{pos:04X})")
            print(f"对应角度: {angle:.2f} 度")
            print(f"------------------------------")
        else:
            print(f"读取失败，未收到电机 ID {target_id} 的有效响应。")
        
        # 读取角度限制
        print(f"\n正在读取角度限制...")
        min_limit = read_min_angle_limit(ser, target_id)
        max_limit = read_max_angle_limit(ser, target_id)
        
        if min_limit is not None:
            min_angle = (min_limit / 4095.0) * 200.0
            print(f"最小角度限制: {min_limit} (0x{min_limit:04X}) = {min_angle:.2f} 度")
        else:
            print(f"读取最小角度限制失败")
        
        if max_limit is not None:
            max_angle = (max_limit / 4095.0) * 200.0
            print(f"最大角度限制: {max_limit} (0x{max_limit:04X}) = {max_angle:.2f} 度")
        else:
            print(f"读取最大角度限制失败")
            
        ser.close()
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
