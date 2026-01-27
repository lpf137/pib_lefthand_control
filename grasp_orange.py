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

rightHand = [0]*12  # 舵机位置列表，索引0-11对应舵机ID

# 定义伸直动作
def stretch_hand():
    """伸直动作"""
    rightHand[2] = 423
    rightHand[4] = 596
    rightHand[6] = 329
    rightHand[8] = 810
    rightHand[9] = 339
    rightHand[10] = 633
    rightHand[11] = 495
    
    print("执行伸直动作...")
    execute_action([2, 4, 6, 8, 9, 10, 11])

# 定义抓握动作
def grasp_hand():
    """抓握动作"""
    rightHand[9] = 268
    rightHand[10] = 1013
    rightHand[8] = 250
    rightHand[6] = 990
    rightHand[4] = 82
    rightHand[2] = 1018
    
    print("执行抓握动作...")
    execute_action([9,2,10,4,6,8])

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

# 主程序 - 直接执行抓握动作
try:
    print("\n" + "="*40)
    print("灵巧手抓握程序")
    print("端口: COM7, 波特率: 115200")
    print("="*40)
    
    # 直接执行抓握动作
    grasp_hand()
    
    print("\n抓握动作已完成")

except KeyboardInterrupt:
    print("\n程序被中断")
finally:
    portHandler.closePort()
    print("端口已关闭")