# main.py
from scservo_driver import SCServoDriver
import time

# 1. 实例化驱动 (根据实际情况修改端口号)
# Windows: 'COMx', Linux/Mac: '/dev/ttyUSB0' 或 '/dev/tty.usbserial-xxx'
driver = SCServoDriver(port='COM8', baudrate=115200)

try:
    print("开始控制...")

    # === 参数配置 ===
    # 电机列表：ID和目标角度 (按ID从小到大排序)
    motor_commands = [
        (0x4, 115),  # ID 0x12 -> 100度
        (0x12, 75),  # ID 0x11 -> 100度
        (0x11, 75),   # ID 0x8 -> 100度
        (0x4, 120),
        (0x11, 65),
        # (0x7, 0),     # ID 0x7 -> 0度
        # (0x4, 100),   # ID 0x4 -> 100度
        
    ]
    target_time_ms = 1000  # 运行时间 (毫秒)
    target_speed = 1000    # 运行速度
    # ================

    # 依次控制每个电机
    for target_id, target_angle in motor_commands:
        target_pos = int((target_angle / 200.0) * 4095)
        
        # 确保扭矩开启
        print(f"开启扭矩 ID 0x{target_id:X}...")
        driver.set_torque_enable(target_id, 1)
        time.sleep(0.5)

        print(f"指令发送: 电机 0x{target_id:X} -> {target_angle}度 (位置 {target_pos}), 耗时 {target_time_ms}ms, 速度 {target_speed}")
        
        # 发送控制指令
        driver.set_position(servo_id=target_id, position=target_pos, time_ms=target_time_ms, speed=target_speed)
        
        # 等待运动完成 (运行时间 + 缓冲)
        wait_time = (target_time_ms / 1000.0) + 0.5
        time.sleep(wait_time)

        # 读取位置验证
        pos = driver.read_position(target_id)
        if pos is not None:
            current_angle = (pos / 4095.0) * 200.0
            print(f"ID 0x{target_id:X} 当前位置: {pos} (约 {current_angle:.2f}度)")
        else:
            print(f"ID 0x{target_id:X} 位置读取失败")
        
        print("-" * 50)

    print("所有电机控制完成")

except Exception as e:
    print(f"发生错误: {e}")
finally:
    driver.close()