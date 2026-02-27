import serial
import time

class SCServoDriver:
    def __init__(self, port, baudrate=115200, timeout=0.5):
        """
        初始化串口
        :param port: 串口号 (例如 Windows下 'COM3', Linux下 '/dev/ttyUSB0')
        :param baudrate: 波特率 (文档默认为 1000000)
        """
        self.serial = serial.Serial(port, baudrate, timeout=timeout)
        self.header = [0x12, 0x4C] # 字头

    def _calc_checksum(self, id, length, instruction, params):
        """
        计算校验和: ~(ID + Length + Instruction + Parameter1 + ... ParameterN)
        """
        total = id + length + instruction + sum(params)
        return (~total) & 0xFF

    def _send_packet(self, id, instruction, params):
        """
        发送指令包
        """
        length = len(params) + 2
        checksum = self._calc_checksum(id, length, instruction, params)
        
        packet = self.header + [id, length, instruction] + params + [checksum]
        self.serial.write(bytearray(packet))
        
        # 调试用：打印发送的十六进制数据
        print("发送:", ' '.join([f'{x:02X}' for x in packet]))

    def _read_packet(self, expected_len=6):
        """
        读取应答包
        """
        # 直接读取指定长度，依赖 timeout 等待数据
        data = self.serial.read(expected_len)
        if len(data) == expected_len:
            print("接收:", ' '.join([f'{x:02X}' for x in data]))
            return list(data)
        else:
            print(f"接收失败: 期望 {expected_len} 字节, 实际收到 {len(data)} 字节")
            if len(data) > 0:
                print("部分数据:", ' '.join([f'{x:02X}' for x in data]))
        return None

    def close(self):
        self.serial.close()

    # ================= 核心功能函数 =================

    def set_torque_enable(self, servo_id, enable):
        """
        设置扭矩开关 (指令 0x03 WRITE DATA)
        :param servo_id: 舵机ID
        :param enable: True/1=开启扭矩(上电), False/0=关闭扭矩(卸力)
        """
        val = 1 if enable else 0
        # 地址 0x28 (40) 是扭矩开关
        self._send_packet(servo_id, 0x03, [0x28, val])

    def set_position(self, servo_id, position, time_ms=0, speed=0):
        """
        控制单个舵机移动 (指令 0x03 WRITE DATA)
        :param servo_id: 舵机ID (1-253)
        :param position: 目标位置 (0-4095)
        :param time_ms: 运行时间 (ms)，0表示最快。文档推荐使用时间控制平滑度。
        :param speed: 运行速度 (文档指出可能无效，但保留占位)
        """
        # 限制位置范围
        position = max(0, min(4095, position))
        
        # 拆分数据为高低字节
        pos_h, pos_l = (position >> 8) & 0xFF, position & 0xFF
        time_h, time_l = (time_ms >> 8) & 0xFF, time_ms & 0xFF
        spd_h, spd_l = (speed >> 8) & 0xFF, speed & 0xFF
        
        # 写入地址 0x2A (42)，连续写入6个字节：位置(2)+时间(2)+速度(2)
        # 参数: 首地址, 数据...
        params = [0x2A, pos_h, pos_l, time_h, time_l, spd_h, spd_l]
        
        self._send_packet(servo_id, 0x03, params)

    def sync_write_positions(self, servo_data_list, time_ms=0, speed=0):
        """
        同步控制多个舵机 (指令 0x83 SYNC WRITE) - 推荐用于机械臂/多足机器人
        :param servo_data_list: 列表，包含元组 (id, position)。 例如: [(1, 2048), (2, 1000)]
        :param time_ms: 所有舵机的统一运行时间
        :param speed: 统一运行速度
        """
        if not servo_data_list:
            return

        # 构造参数
        # 参数1: 写入首地址 0x2A
        # 参数2: 每个舵机的数据长度 L = 6 (位置2+时间2+速度2)
        params = [0x2A, 0x06]
        
        time_h, time_l = (time_ms >> 8) & 0xFF, time_ms & 0xFF
        spd_h, spd_l = (speed >> 8) & 0xFF, speed & 0xFF

        for servo_id, position in servo_data_list:
            position = max(0, min(4095, position))
            pos_h, pos_l = (position >> 8) & 0xFF, position & 0xFF
            
            # 每个舵机的数据块: ID + P_H + P_L + T_H + T_L + S_H + S_L
            params.append(servo_id)
            params.extend([pos_h, pos_l, time_h, time_l, spd_h, spd_l])

        # 发送广播 ID 0xFE (254)
        self._send_packet(0xFE, 0x83, params)

    def read_position(self, servo_id):
        """
        读取舵机当前位置 (指令 0x02 READ DATA)
        :return: 当前位置数值 (0-4095) 或 -1 (读取失败)
        """
        # 清空缓冲区，防止读到旧数据
        self.serial.reset_input_buffer()
        
        # 读取地址 0x38 (56)，长度 2个字节
        params = [0x38, 0x02]
        self._send_packet(servo_id, 0x02, params)
        
        # 读取返回包 (Header 2 + ID 1 + Len 1 + Err 1 + Param 2 + Sum 1 = 8 bytes)
        response = self._read_packet(8)
        
        if response and len(response) == 8:
            # 简单校验字头
            if response[0] == 0x12 and response[1] == 0x4C:
                # 参数在 index 5 和 6 (高位在前)
                pos_h = response[5]
                pos_l = response[6]
                return (pos_h << 8) | pos_l
        
        return -1

    def read_memory(self, servo_id, address, length):
        """
        读取舵机内存表
        :param servo_id: 舵机ID
        :param address: 内存起始地址
        :param length: 读取长度
        :return: 字节列表 或 None
        """
        self.serial.reset_input_buffer()
        params = [address, length]
        self._send_packet(servo_id, 0x02, params)
        
        # 返回包长度: Header(2)+ID(1)+Len(1)+Err(1)+Param(length)+Sum(1) = 6 + length
        expected_len = 6 + length
        response = self._read_packet(expected_len)
        
        if response and len(response) == expected_len:
             if response[0] == 0x12 and response[1] == 0x4C:
                 # 参数从 index 5 开始
                 return response[5:5+length]
        return None