#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from scservo_driver import SCServoDriver
import time
from collections import deque
import sys
import os

# 添加scservo_sdk路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scservo_sdk import *

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 30

parser = argparse.ArgumentParser()
parser.add_argument(
    "--depthSource", type=str, default="stereo", choices=["stereo", "neural"]
)
args = parser.parse_args()

modelDescription = dai.NNModelDescription("yolov6-nano")
size = (640, 400)

if args.depthSource == "stereo":
    fps = STEREO_DEFAULT_FPS
else:
    fps = NEURAL_FPS

# 初始化电机驱动
driver = SCServoDriver(port='COM8', baudrate=115200)

class HandController:
    def __init__(self, port='COM7', baudrate=115200):
        self.portHandler = PortHandler(port)
        self.packetHandler = scscl(self.portHandler)
        self.rightHand = [0]*12
        
        if self.portHandler.openPort():
            print(f"成功打开灵巧手端口 {port}")
        else:
            print(f"打开灵巧手端口 {port} 失败")

        if self.portHandler.setBaudRate(baudrate):
            print(f"成功设置灵巧手波特率 {baudrate}")
        else:
            print("设置灵巧手波特率失败")

    def execute_action(self, id_list):
        for id in id_list:
            scs_comm_result, scs_error = self.packetHandler.WritePos(id, self.rightHand[id], 0, 1500)
            if scs_comm_result != COMM_SUCCESS:
                print("通信错误 ID %d: %s" % (id, self.packetHandler.getTxRxResult(scs_comm_result)))
            if scs_error != 0:
                print("舵机错误 ID %d: %s" % (id, self.packetHandler.getRxPacketError(scs_error)))
            time.sleep(0.5)
        print("动作完成")

    def stretch_hand(self):
        """伸直动作"""
        self.rightHand[2] = 423
        self.rightHand[4] = 596
        self.rightHand[6] = 329
        self.rightHand[8] = 810
        self.rightHand[9] = 339
        self.rightHand[10] = 633
        self.rightHand[11] = 495
        
        print("执行伸直动作...")
        #self.execute_action([2, 4, 6, 8, 9, 10, 11])
        self.execute_action([10, 9, 8, 6, 4, 2, 11])

    def grasp_hand(self):
        """抓握动作"""
        self.rightHand[10] = 1000
        self.rightHand[8] = 250
        self.rightHand[6] = 990
        self.rightHand[4] = 100
        self.rightHand[2] = 980
        
        print("执行抓握动作...")
        self.execute_action([8, 10, 6, 4, 2])
        
    def close(self):
        self.portHandler.closePort()

# 初始化灵巧手
hand_controller = HandController()

class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
        # 只识别这些标签
        self.target_labels = ["bottle", "orange"]
        # 置信度阈值
        self.confidence_threshold = 0.3
        
        # X坐标缓冲区用于平均值计算
        self.x_buffer = deque(maxlen=10)  # 保存最近10个x值
        
        # 电机0x8映射参数: x=-198 -> angle=100, x=90 -> angle=60
        self.x_min_motor8 = -209
        self.x_max_motor8 = 55
        self.angle_at_x_min_motor8 = 100  # x=-198时的角度
        self.angle_at_x_max_motor8 = 60   # x=55时的角度
        
        # 电机0x4映射参数: x=-189 -> angle=100, x=70 -> angle=125
        self.x_min_motor4 = -189
        self.x_max_motor4 = 70
        self.angle_at_x_min_motor4 = 100  # x=-189时的角度
        self.angle_at_x_max_motor4 = 125  # x=70时的角度
        
        # 电机0x11映射参数: x=-198 -> angle=100, x=73 -> angle=78
        self.x_min_motor11 = -198
        self.x_max_motor11 = 73
        self.angle_at_x_min_motor11 = 100  # x=-198时的角度
        self.angle_at_x_max_motor11 = 78   # x=73时的角度
        
        # 电机参数
        self.motor_id_8 = 0x8
        self.motor_id_4 = 0x4
        self.motor_id_11 = 0x11
        self.motor_id_12 = 0x12
        self.target_time_ms = 800  # 更快的响应时间
        self.target_speed = 1000
        
        # 上次发送的角度，用于减少不必要的指令
        self.last_angle_motor8 = None
        self.last_angle_motor4 = None
        self.last_angle_motor11 = None
        self.angle_threshold = 1.0  # 角度变化阈值（度）
        
        # 限制发送频率
        self.last_send_time = 0
        self.send_interval = 0.3  # 最小发送间隔（秒）
        
        # 开启电机扭矩
        driver.set_torque_enable(self.motor_id_8, 1)
        time.sleep(0.05)
        driver.set_torque_enable(self.motor_id_4, 1)
        time.sleep(0.05)
        driver.set_torque_enable(self.motor_id_11, 1)
        time.sleep(0.05)
        driver.set_torque_enable(self.motor_id_12, 1)
        time.sleep(0.5)
        
        # 抓取控制变量
        self.last_bottle_x = None
        self.bottle_stable_start_time = 0
        self.is_grasping = False
        
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb)

    def map_x_to_angle_motor8(self, x):
        """将x坐标线性映射到电机0x8的角度
        x=-109 -> angle=100
        x=61 -> angle=60
        """
        # 限制x范围
        x = max(self.x_min_motor8, min(self.x_max_motor8, x))
        
        # 线性映射: angle = angle_at_x_min + (x - x_min) * (angle_at_x_max - angle_at_x_min) / (x_max - x_min)
        angle = self.angle_at_x_min_motor8 + (x - self.x_min_motor8) * (self.angle_at_x_max_motor8 - self.angle_at_x_min_motor8) / (self.x_max_motor8 - self.x_min_motor8)
        
        # 限制角度范围
        angle = max(min(self.angle_at_x_min_motor8, self.angle_at_x_max_motor8), 
                   min(max(self.angle_at_x_min_motor8, self.angle_at_x_max_motor8), angle))
        return angle
    
    def map_x_to_angle_motor4(self, x):
        """将x坐标线性映射到电机0x4的角度
        x=-189 -> angle=100
        x=-56 -> angle=105
        """
        # 限制x范围
        x = max(self.x_min_motor4, min(self.x_max_motor4, x))
        
        # 线性映射
        angle = self.angle_at_x_min_motor4 + (x - self.x_min_motor4) * (self.angle_at_x_max_motor4 - self.angle_at_x_min_motor4) / (self.x_max_motor4 - self.x_min_motor4)
        
        # 限制角度范围
        angle = max(min(self.angle_at_x_min_motor4, self.angle_at_x_max_motor4), 
                   min(max(self.angle_at_x_min_motor4, self.angle_at_x_max_motor4), angle))
        return angle
    
    def map_x_to_angle_motor11(self, x):
        """将x坐标线性映射到电机0x11的角度
        x=-198 -> angle=100
        x=73 -> angle=78
        """
        # 限制x范围
        x = max(self.x_min_motor11, min(self.x_max_motor11, x))
        
        # 线性映射
        angle = self.angle_at_x_min_motor11 + (x - self.x_min_motor11) * (self.angle_at_x_max_motor11 - self.angle_at_x_min_motor11) / (self.x_max_motor11 - self.x_min_motor11)
        
        # 限制角度范围
        angle = max(min(self.angle_at_x_min_motor11, self.angle_at_x_max_motor11), 
                   min(max(self.angle_at_x_min_motor11, self.angle_at_x_max_motor11), angle))
        return angle

    def process(self, depthPreview, detections, rgbPreview):
        rgbPreview = rgbPreview.getCvFrame()
        # 过滤检测结果
        filtered_detections = [
            d for d in detections.detections 
            if d.labelName in self.target_labels and d.confidence >= self.confidence_threshold
        ]
        
        # 如果检测到目标物体，使用第一个检测结果的x坐标
        if filtered_detections:
            x_coord = int(filtered_detections[0].spatialCoordinates.x)

            # 抓取逻辑
            if filtered_detections[0].labelName == "bottle":
                if self.last_bottle_x is None:
                    self.last_bottle_x = x_coord
                    self.bottle_stable_start_time = time.time()
                else:
                    # 检查是否静止
                    if abs(x_coord - self.last_bottle_x) < 20:
                        if time.time() - self.bottle_stable_start_time > 3.0:
                            if not self.is_grasping:
                                print(f"检测到瓶子静止超过3秒，执行抓取...")
                                
                                # 移动手臂电机到抓取位置
                                # Motor 0x4 -> 140度
                                print("调整手臂电机: 0x4 -> 140°, 0x11 -> 50°")
                                target_pos_4 = int((140 / 200.0) * 4095)
                                driver.set_position(servo_id=self.motor_id_4, position=target_pos_4, 
                                                  time_ms=self.target_time_ms, speed=self.target_speed)
                                time.sleep(0.05)
                                
                                # Motor 0x11 -> 50度
                                target_pos_11 = int((50 / 200.0) * 4095)
                                driver.set_position(servo_id=self.motor_id_11, position=target_pos_11, 
                                                  time_ms=self.target_time_ms, speed=self.target_speed)
                                time.sleep(0.5)

                                # Motor 0x8 -> 当前角度 -10度
                                # 仅在有记录角度时执行
                                if self.last_angle_motor8 is not None:
                                    current_angle = self.last_angle_motor8
                                    new_angle_8 = current_angle - 10.0
                                    print(f"调整电机 0x8: {current_angle:.1f}° -> {new_angle_8:.1f}°")
                                    
                                    target_pos_8 = int((new_angle_8 / 200.0) * 4095)
                                    driver.set_position(servo_id=self.motor_id_8, position=target_pos_8, 
                                                      time_ms=self.target_time_ms, speed=self.target_speed)
                                    self.last_angle_motor8 = new_angle_8
                                else:
                                    print("电机 0x8 没有角度记录，跳过调整")

                                time.sleep(1.5)  # 等待手臂到位
                                
                                print("执行灵巧手抓取...")
                                hand_controller.grasp_hand()
                                
                                # 抓取成功后，手臂归位动作
                                print("手臂归位: 0x11 -> 60°, 0x12 -> 200°")
                                time.sleep(0.5)
                                
                                target_pos_11_home = int((60 / 200.0) * 4095)
                                driver.set_position(servo_id=self.motor_id_11, position=target_pos_11_home, 
                                                  time_ms=self.target_time_ms, speed=self.target_speed)
                                time.sleep(0.05)
                                
                                target_pos_12_home = int((200 / 200.0) * 4095)
                                driver.set_position(servo_id=self.motor_id_12, position=target_pos_12_home, 
                                                  time_ms=self.target_time_ms, speed=self.target_speed)
                                
                                self.is_grasping = True
                    else:
                        # 如发生移动，重置
                        self.last_bottle_x = x_coord
                        self.bottle_stable_start_time = time.time()
            else:
                self.last_bottle_x = None

            self.x_buffer.append(x_coord)
            
            # 计算平均x坐标
            if len(self.x_buffer) > 0:
                current_time = time.time()
                
                # 检查是否到达发送间隔
                if current_time - self.last_send_time < self.send_interval:
                    # 还没到发送时间，只显示但不发送指令
                    avg_x = sum(self.x_buffer) / len(self.x_buffer)
                    target_angle_motor8 = self.map_x_to_angle_motor8(avg_x)
                    target_angle_motor4 = self.map_x_to_angle_motor4(avg_x)
                    target_angle_motor11 = self.map_x_to_angle_motor11(avg_x)
                    
                    cv2.putText(rgbPreview, f"Avg X: {avg_x:.1f} mm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x8: {target_angle_motor8:.1f} deg", 
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x4: {target_angle_motor4:.1f} deg", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x11: {target_angle_motor11:.1f} deg", 
                               (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # 到达发送时间，计算并发送指令
                    if hasattr(self, 'is_grasping') and self.is_grasping:
                        # 抓取模式下，跳过视觉伺服控制
                        pass
                    else:
                        avg_x = sum(self.x_buffer) / len(self.x_buffer)
                        
                        # 计算三个电机的目标角度
                        target_angle_motor8 = self.map_x_to_angle_motor8(avg_x)
                        target_angle_motor4 = self.map_x_to_angle_motor4(avg_x)
                        target_angle_motor11 = self.map_x_to_angle_motor11(avg_x)
                        
                        # 检查角度变化是否足够大
                        send_motor8 = (self.last_angle_motor8 is None or 
                                      abs(target_angle_motor8 - self.last_angle_motor8) >= self.angle_threshold)
                        send_motor4 = (self.last_angle_motor4 is None or 
                                      abs(target_angle_motor4 - self.last_angle_motor4) >= self.angle_threshold)
                        send_motor11 = (self.last_angle_motor11 is None or 
                                      abs(target_angle_motor11 - self.last_angle_motor11) >= self.angle_threshold)
                        
                        # 发送电机0x8控制指令
                        if send_motor8:
                            target_pos_8 = int((target_angle_motor8 / 200.0) * 4095)
                            driver.set_position(servo_id=self.motor_id_8, position=target_pos_8, 
                                              time_ms=self.target_time_ms, speed=self.target_speed)
                            self.last_angle_motor8 = target_angle_motor8
                            time.sleep(0.05)  # 两个电机指令之间添加延迟
                        
                        # 发送电机0x4控制指令
                        if send_motor4:
                            target_pos_4 = int((target_angle_motor4 / 200.0) * 4095)
                            driver.set_position(servo_id=self.motor_id_4, position=target_pos_4, 
                                              time_ms=self.target_time_ms, speed=self.target_speed)
                            self.last_angle_motor4 = target_angle_motor4
                            time.sleep(0.05)  # 电机指令之间添加延迟
                        
                        # 发送电机0x11控制指令
                        if send_motor11:
                            target_pos_11 = int((target_angle_motor11 / 200.0) * 4095)
                            driver.set_position(servo_id=self.motor_id_11, position=target_pos_11, 
                                              time_ms=self.target_time_ms, speed=self.target_speed)
                            self.last_angle_motor11 = target_angle_motor11
                        
                        # 更新最后发送时间
                        if send_motor8 or send_motor4 or send_motor11:
                            self.last_send_time = current_time
                        
                    # 在画面上显示映射信息
                    avg_x = sum(self.x_buffer) / len(self.x_buffer)
                    target_angle_motor8 = self.map_x_to_angle_motor8(avg_x)
                    if hasattr(self, 'is_grasping') and self.is_grasping:
                        # 显示抓取状态
                         cv2.putText(rgbPreview, "Status: GRASPING", 
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        target_angle_motor4 = self.map_x_to_angle_motor4(avg_x)
                        target_angle_motor11 = self.map_x_to_angle_motor11(avg_x)
                        cv2.putText(rgbPreview, f"Motor 0x4: {target_angle_motor4:.1f} deg", 
                                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(rgbPreview, f"Motor 0x11: {target_angle_motor11:.1f} deg", 
                                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                    cv2.putText(rgbPreview, f"Avg X: {avg_x:.1f} mm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x8: {target_angle_motor8:.1f} deg", 
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                              
        
        self.displayResults(rgbPreview, filtered_detections)

    def displayResults(self, rgbFrame, detections):
        height, width, _ = rgbFrame.shape
        for detection in detections:
            self.drawDetections(rgbFrame, detection, width, height)

        cv2.imshow("rgb", rgbFrame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.stopPipeline()
        elif key == ord('s'):
            print("按下S键，释放灵巧手并复位手臂...")
            hand_controller.stretch_hand()
            
            # 手臂关节初始化复位
            print("手臂关节复位到初始位置...")
            motor_commands = [
                (0x12, 100),  # ID 0x12 -> 100度
                (0x11, 100),  # ID 0x11 -> 100度
                (0x8, 100),   # ID 0x8 -> 100度
                (0x7, 0),     # ID 0x7 -> 0度
                (0x4, 100),   # ID 0x4 -> 100度
            ]
            
            for motor_id, angle in motor_commands:
                target_pos = int((angle / 200.0) * 4095)
                driver.set_position(servo_id=motor_id, position=target_pos, 
                                  time_ms=self.target_time_ms, speed=self.target_speed)
                time.sleep(0.05)
            
            print("复位完成")
            self.is_grasping = False
            self.last_bottle_x = None

    def drawDetections(self, frame, detection, frameWidth, frameHeight):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        color = (255, 255, 255)
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

try:
    print("开始视觉控制系统...")
    print(f"电机0x8映射: X=-205mm -> 105°, X=61mm -> 65°")
    print(f"电机0x4映射: X=-189mm -> 100°, X=-56mm -> 105°")

    
    # Creates the pipeline and a default device implicitly
    with dai.Pipeline() as p:
        # Define sources and outputs
        platform = p.getDefaultDevice().getPlatform()

        camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)
        monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
        monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
        if args.depthSource == "stereo":
            depthSource = p.create(dai.node.StereoDepth)
            depthSource.setExtendedDisparity(True)
            if platform == dai.Platform.RVC2:
                depthSource.setOutputSize(640, 400)
            monoLeft.requestOutput(size).link(depthSource.left)
            monoRight.requestOutput(size).link(depthSource.right)
        elif args.depthSource == "neural":
            depthSource = p.create(dai.node.NeuralDepth).build(
                monoLeft.requestFullResolutionOutput(),
                monoRight.requestFullResolutionOutput(),
                dai.DeviceModelZoo.NEURAL_DEPTH_LARGE,
            )
        else:
            raise ValueError(f"Invalid depth source: {args.depthSource}")

        spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
            camRgb, depthSource, modelDescription
        )
        visualizer = p.create(SpatialVisualizer)

        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        visualizer.build(
            spatialDetectionNetwork.passthroughDepth,
            spatialDetectionNetwork.out,
            spatialDetectionNetwork.passthrough,
        )

        print("Starting pipeline with depth source: ", args.depthSource)

        p.run()

except Exception as e:
    print(f"发生错误: {e}")
finally:
    try:
        driver.close()
    except:
        pass
    try:
        hand_controller.close()
    except:
        pass
    cv2.destroyAllWindows()
    print("系统已关闭")