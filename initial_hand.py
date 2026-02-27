#!/usr/bin/env python3

import sys
import os
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

# 定义伸直动作
def stretch_hand():
    """伸直动作"""
    rightHand[1] = 491
    rightHand[2] = 460
    rightHand[3] = 474
    rightHand[4] = 234
    rightHand[5] = 491
    rightHand[6] = 143
    rightHand[7] = 474
    rightHand[8] = 390
    rightHand[9] = 271
    rightHand[10] = 511
    rightHand[11] = 406
    rightHand[21] = 1021
    rightHand[22] = 423
    rightHand[23] = 0
    rightHand[24] = 596
    rightHand[25] = 1022
    rightHand[26] = 329
    rightHand[27] = 74
    rightHand[28] = 810
    rightHand[29] = 576
    rightHand[30] = 633
    rightHand[31] = 495
    #rightHand舵机数字加20就是lefthand

    
    print("执行伸直动作...")
    execute_action([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21])


# 定义抓握动作
def grasp_hand():
    """抓握动作"""
    rightHand[10] = 1000
    rightHand[8] = 300
    rightHand[6] = 990
    rightHand[4] = 130
    rightHand[2] = 900
    
    print("执行抓握动作...")
    execute_action([10, 8, 6, 4, 2])

# 执行动作
def execute_action(id_list):
    """向指定ID列表的舵机发送位置指令"""
    for id in id_list:
        scs_comm_result, scs_error = packetHandler.WritePos(id, rightHand[id], 0, 1500)
        if scs_comm_result != COMM_SUCCESS:
            print("通信错误 ID %d: %s" % (id, packetHandler.getTxRxResult(scs_comm_result)))
        if scs_error != 0:
            print("舵机错误 ID %d: %s" % (id, packetHandler.getRxPacketError(scs_error)))
        time.sleep(0.5)  # 等待舵机运动
    print("动作完成")

# 主程序 - 直接执行伸直动作
try:
    print("\n" + "="*40)
    print("灵巧手伸直程序")
    print("端口: COM7, 波特率: 115200")
    print("="*40)
    
    # 直接执行伸直动作
    stretch_hand()
    
    print("\n伸直动作已完成")

except KeyboardInterrupt:
    print("\n程序被中断")
finally:
    portHandler.closePort()
    print("端口已关闭")