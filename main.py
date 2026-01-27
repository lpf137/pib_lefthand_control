# main.py
from scservo_driver import SCServoDriver
import time

# 1. 实例化驱动 (根据实际情况修改端口号)
# Windows: 'COMx', Linux/Mac: '/dev/ttyUSB0' 或 '/dev/tty.usbserial-xxx'
driver = SCServoDriver(port='COM8', baudrate=115200)

try:
    print("开始控制...")

    # === 参数配置 ===
    target_id = 0x8       # 目标电机 ID (十六进制0x8 = 十进制8)
    target_angle = 60  # 目标角度 (0-200度)
    target_time_ms = 1000  # 运行时间 (毫秒)，时间越长越慢
    target_speed = 1000    # 运行速度 (某些电机需要此参数)
    # ================

    target_pos = int((target_angle / 200.0) * 4095)
    
    # 确保扭矩开启
    print(f"开启扭矩 ID {target_id}...")
    driver.set_torque_enable(target_id, 1)
    time.sleep(0.1)

    print(f"指令发送: 电机 {target_id} -> {target_angle}度 (位置 {target_pos}), 耗时 {target_time_ms}ms, 速度 {target_speed}")
    
    # 发送控制指令
    driver.set_position(servo_id=target_id, position=target_pos, time_ms=target_time_ms, speed=target_speed)
    
    # 等待运动完成 (运行时间 + 缓冲)
    wait_time = (target_time_ms / 1000.0) + 0.5
    time.sleep(wait_time)

    # 读取位置验证
    pos = driver.read_position(target_id)
    if pos is not None:
        current_angle = (pos / 4095.0) * 200.0
        print(f"ID {target_id} 当前位置: {pos} (约 {current_angle:.2f}度)")
    else:
        print(f"ID {target_id} 位置读取失败")

    print("控制完成")

except Exception as e:
    print(f"发生错误: {e}")
finally:
    driver.close()