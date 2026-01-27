class NECDecoder:
    def __init__(self):
        pass

    def decode(self, pulses):
        """
        pulses: [(pulse_length), ...] in microseconds
        Returns: (address, command) or None
        """

        # NEC 协议典型参数（不同遥控器有一点点漂移没事）
        LEADER_HIGH = 9000
        LEADER_LOW  = 4500
        BIT_HIGH    = 560
        BIT_0_LOW   = 560
        BIT_1_LOW   = 1690

        # 1. 校验前导码
        if len(pulses) < 2:
            return None
        
        if not (7000 < pulses[0] < 10000):
            return None
        if not (3500 < pulses[1] < 5500):
            return None

        # 2. 开始解析 32 位数据
        bits = []
        i = 2  # 从第三个脉冲开始

        while i < len(pulses) - 1:
            high = pulses[i]
            low = pulses[i+1]

            # 高电平应该接近 560us
            if not (300 < high < 900):
                return None

            # 判断 0 或 1
            if 300 < low < 900:
                bits.append(0)
            elif 1500 < low < 2300:
                bits.append(1)
            else:
                return None

            i += 2

        if len(bits) < 32:
            return None
        
        # 3. 解析 address/command
        address = self.bits_to_byte(bits[0:8])
        command = self.bits_to_byte(bits[16:24])

        return address, command

    def bits_to_byte(self, bits):
        val = 0
        for i, b in enumerate(bits):
            val |= (b << i)
        return val
