import pigpio
import time

pi = pigpio.pi()  # 连接daemon
IR_PIN = 18  # BCM 18

pi.set_mode(IR_PIN, pigpio.INPUT)
pi.set_pull_up_down(IR_PIN, pigpio.PUD_UP)

# NEC协议阈值（微秒）
LEAD_LOW_MIN = 8500   # 引导头低电平 ~9ms
LEAD_LOW_MAX = 9500
LEAD_HIGH_MIN = 4000  # 引导头高电平 ~4.5ms
LEAD_HIGH_MAX = 5000
BIT_LOW = 562         # 逻辑0/1 低电平固定 ~562us
BIT_0_HIGH_MAX = 800  # 逻辑0 高电平短
BIT_1_HIGH_MIN = 1400 # 逻辑1 高电平长

key_map = {
    0x45: "1", 0x46: "2", 0x47: "3",
    0x44: "4", 0x40: "5", 0x43: "6",
    0x07: "7", 0x15: "8", 0x09: "9",
    0x16: "*", 0x19: "0", 0x0D: "#",
    0x18: "up", 0x52: "down",
    0x08: "left", 0x5A: "right",
    0x1C: "OK"
}

def decode_nec():
    # 等信号开始（低电平）
    start_time = time.time()
    while pi.read(IR_PIN) == 1:
        if time.time() - start_time > 1:  # 超时1秒
            return None

    # 测量引导头低电平
    low_start = time.time()
    while pi.read(IR_PIN) == 0:
        pass
    low_us = int((time.time() - low_start) * 1000000)

    # 测量引导头高电平
    high_start = time.time()
    while pi.read(IR_PIN) == 1:
        pass
    high_us = int((time.time() - high_start) * 1000000)

    if not (LEAD_LOW_MIN < low_us < LEAD_LOW_MAX and LEAD_HIGH_MIN < high_us < LEAD_HIGH_MAX):
        return None  # 不是NEC引导头

    # 读32位数据
    data = 0
    for bit in range(32):
        # 低电平（固定）
        while pi.read(IR_PIN) == 0:
            pass
        # 高电平（区分0/1）
        high_start = time.time()
        while pi.read(IR_PIN) == 1:
            pass
        high_us = int((time.time() - high_start) * 1000000)

        if high_us > BIT_1_HIGH_MIN:
            data |= (1 << (31 - bit))  # 逻辑1
        # else 逻辑0

    # 校验地址和命令
    addr = data & 0xFF
    addr_inv = (data >> 8) & 0xFF
    cmd = (data >> 16) & 0xFF
    cmd_inv = (data >> 24) & 0xFF

    if addr + addr_inv == 0xFF and cmd + cmd_inv == 0xFF:
        return cmd
    return None

print("pigpio NEC解码监听中... 按遥控器试试")

while True:
    cmd = decode_nec()
    if cmd is not None:
        key = key_map.get(cmd, f"未知(0x{cmd:02X})")
        print(f"你按下了： {key} (代码: 0x{cmd:02X})")
    time.sleep(0.1)