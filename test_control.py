"""Joint test script for arm (SCServo bus) and dexterous hand (vendor SDK)."""

from arm_scservo_driver import SCServoDriver
import os
import sys
import time


# 添加scservo_sdk路径
sys.path.append(os.path.dirname(__file__))
from scservo_sdk import *

# 初始化 PortHandler
portHandler = PortHandler('COM7')  # 使用COM7端口

# 初始化 PacketHandler
packetHandler = scscl(portHandler)
    
# 打开端口
if portHandler.openPort():
    print("成功打开端口 COM7")
else:
    print("打开端口失败")
    quit()

# 设置波特率 115200
if portHandler.setBaudRate(115200):
    print("成功设置波特率 115200")
else:
    print("设置波特率失败")
    quit()

rightHand = [0]*32  # 覆盖右手与左手镜像所需的全部舵机ID

def ready_grasp_hand():
    rightHand[10] = 50
    rightHand[9] = 579
    rightHand[8] = 894
    rightHand[6] = 784
    rightHand[4] = 1000
    rightHand[2] = 1000
    rightHand[11] = 315

    execute_action([10])


# 定义抓握动作
def grasp_hand():
    """抓握动作"""
    rightHand[10] = 100
    rightHand[9] = 579
    rightHand[8] = 894
    rightHand[6] = 784
    rightHand[4] = 1000
    rightHand[2] = 1000
    rightHand[11] = 350


    
    print("执行抓握动作...")
    execute_action([11,9,10,8,6])
# 执行动作
def execute_action(id_list):
    """向指定ID列表的舵机发送位置指令"""
    for id in id_list:
        scs_comm_result, scs_error = packetHandler.WritePos(id, rightHand[id], 0, 1500)
        if scs_comm_result != COMM_SUCCESS:
            print("通信错误 ID %d: %s" % (id, packetHandler.getTxRxResult(scs_comm_result)))
        if scs_error != 0:
            print("舵机错误 ID %d: %s" % (id, packetHandler.getRxPacketError(scs_error)))
        time.sleep(1)  # 等待舵机运动
    print("动作完成")

# 机械臂驱动 (COM8)
arm_driver = SCServoDriver(port='COM8', baudrate=115200)

try:
    print("开始控制...")

    # === 参数配置 ===
    # 电机列表：ID和目标角度 (按ID从小到大排序)
    motor_commands = [

        (0x6, 67),  # ID 0x9 -> 115度
        (0x10, 20),  # ID 0x12 -> 100度
        (0x3, 85),
        (0x9, 88),
        
        # (0x6, 65),   # ID 0x8 -> 100度
        # (0x9, 115),
        # (0x10, 0),  # ID 0x12 -> 100度
        # (0x6, 77),  # ID 0x11 -> 100度
        # (0x9, 92),   # ID 0x8 -> 100度
        # (0x10, 130),
        # (0x4, 120),
        # (0x11, 65),
        # (0x7, 0),     # ID 0x7 -> 0度
        # (0x4, 100),   # ID 0x4 -> 100度
        
    ]
    target_time_ms = 1000  # 运行时间 (毫秒)
    target_speed = 1000    # 运行速度
    hand_action = "stretch"  # 可选: "stretch", "grasp" 或 None
    hand_time_ms = 3000
    hand_delay_s = 1.0
    # ================

    # 依次控制每个电机
    for target_id, target_angle in motor_commands:
        target_pos = int((target_angle / 200.0) * 4095)
        
        # 确保扭矩开启
        print(f"开启扭矩 ID 0x{target_id:X}...")
        arm_driver.set_torque_enable(target_id, 1)
        time.sleep(0.5)

        print(f"指令发送: 电机 0x{target_id:X} -> {target_angle}度 (位置 {target_pos}), 耗时 {target_time_ms}ms, 速度 {target_speed}")
        
        # 发送控制指令
        arm_driver.set_position(servo_id=target_id, position=target_pos, time_ms=target_time_ms, speed=target_speed)
        
        # 等待运动完成 (运行时间 + 缓冲)
        wait_time = (target_time_ms / 1000.0) + 0.5
        time.sleep(wait_time)

        # 读取位置验证
        pos = arm_driver.read_position(target_id)
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
    arm_driver.close()

# 主程序 - 直接执行伸直动作
try:
    print("\n" + "="*40)
    print("灵巧手伸直程序")
    print("端口: COM7, 波特率: 115200")
    print("="*40)
    
    # 直接执行伸直动作
    ready_grasp_hand()
    time.sleep(1)
    grasp_hand()
    
    print("\n伸直动作已完成")

except KeyboardInterrupt:
    print("\n程序被中断")
finally:
    portHandler.closePort()
    print("端口已关闭")
