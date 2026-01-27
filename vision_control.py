#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from scservo_driver import SCServoDriver
import time
from collections import deque

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
        self.x_min_motor8 = -198
        self.x_max_motor8 = 90
        self.angle_at_x_min_motor8 = 100  # x=-198时的角度
        self.angle_at_x_max_motor8 = 60   # x=90时的角度
        
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
        time.sleep(0.5)
        
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
                    cv2.putText(rgbPreview, f"Avg X: {avg_x:.1f} mm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x8: {target_angle_motor8:.1f} deg", 
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x4: {target_angle_motor4:.1f} deg", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgbPreview, f"Motor 0x11: {target_angle_motor11:.1f} deg", 
                               (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                              
        
        self.displayResults(rgbPreview, filtered_detections)

    def displayResults(self, rgbFrame, detections):
        height, width, _ = rgbFrame.shape
        for detection in detections:
            self.drawDetections(rgbFrame, detection, width, height)

        cv2.imshow("rgb", rgbFrame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

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
    driver.close()
    cv2.destroyAllWindows()
    print("系统已关闭")