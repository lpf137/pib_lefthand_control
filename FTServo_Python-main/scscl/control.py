#!/usr/bin/env python

# 初始化
import sys
import os

sys.path.append("..")
from scservo_sdk import *                      # Uses FTServo SDK library


# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler('/dev/ttyUSB0')# ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"


# Initialize PacketHandler instance
# Get methods and members of Protocol
packetHandler = scscl(portHandler)
    
# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate 1000000
if portHandler.setBaudRate(115200):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()



rightHand = [0]*12  # 右手舵机位置列表，索引1-11对应舵机ID
# 定义伸直动作
def fun1():
    rightHand[1] = 491
    rightHand[2] = 349
    rightHand[3] = 474
    rightHand[4] = 172
    rightHand[5] = 491
    rightHand[6] = 143
    rightHand[7] = 474
    rightHand[8] = 381
    rightHand[9] = 605
    rightHand[10] = 638
    rightHand[11] = 406

# 定义握手动作
def fun2():
    rightHand[1] = 491
    rightHand[2] = 566
    rightHand[3] = 474
    rightHand[4] = 542
    rightHand[5] = 491
    rightHand[6] = 512
    rightHand[7] = 474
    rightHand[8] = 698
    rightHand[9] = 440
    rightHand[10] = 86
    rightHand[11] = 406

# 定义抓取动作
def fun3():
    rightHand[1] = 549
    rightHand[2] = 843
    rightHand[3] = 502
    rightHand[4] = 759
    rightHand[5] = 491
    rightHand[6] = 701
    rightHand[7] = 420
    rightHand[8] = 989
    rightHand[9] = 650
    rightHand[10] = 992
    rightHand[11] = 406


def action():
    for id in range(1, 12):
        scs_comm_result, scs_error = packetHandler.WritePos(id, rightHand[id], 0, 1500)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(scs_comm_result))
        if scs_error != 0:
            print("%s" % packetHandler.getRxPacketError(scs_error))
        time.sleep(((1000-20)/(1500) + 0.05))#[(P1-P0)/(V)] + 0.1

# 主循环，控制逻辑
try:
    while True:

        cmd = input('please input cmd(1-3):')
        while cmd >'3' or cmd <'1' or len(cmd)>1:
            cmd = input('please Reinput cmd(1-3):')

        if cmd =='1':
            # 伸直动作
            fun1()
            action()
        elif cmd =='2':
            # 握手动作
            fun1()
            action()
            fun2()
            action()   
        elif cmd =='3':
            # 抓取动作
            fun1()
            action()
            fun3()
            action()
        

except KeyboardInterrupt:
    portHandler.closePort()




