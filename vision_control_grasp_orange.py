#!/usr/bin/env python3

import argparse
import os
import sys
import time
from collections import deque

import cv2
import depthai as dai

from arm_scservo_driver import SCServoDriver

# 添加 scservo_sdk 路径，用于灵巧手控制
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scservo_sdk import *

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 30

parser = argparse.ArgumentParser()
parser.add_argument("--depthSource", type=str, default="stereo", choices=["stereo", "neural"])
parser.add_argument("--bottleMinConf", type=float, default=0.25, help="Minimum confidence for bottle")
parser.add_argument("--orangeMinConf", type=float, default=0.12, help="Minimum confidence for orange")
parser.add_argument("--ballMinConf", type=float, default=0.25, help="Minimum confidence for ball")
parser.add_argument(
    "--holdMs",
    type=int,
    default=450,
    help="Keep last seen bbox/label for this long when detections flicker",
)
parser.add_argument(
    "--showBelowThreshold",
    action="store_true",
    help="Also draw target detections below threshold (debug)",
)
args = parser.parse_args()

modelDescription = dai.NNModelDescription("yolov6-nano")
size = (640, 400)
fps = STEREO_DEFAULT_FPS if args.depthSource == "stereo" else NEURAL_FPS


def deg_to_position(angle_deg: float) -> int:
    angle_deg = max(0.0, min(200.0, angle_deg))
    return int((angle_deg / 200.0) * 4095)


class HandController:
    def __init__(self, port: str = "COM7", baudrate: int = 115200):
        self.portHandler = PortHandler(port)
        self.packetHandler = scscl(self.portHandler)
        self.rightHand = [0] * 12

        if self.portHandler.openPort():
            print(f"成功打开灵巧手端口 {port}")
        else:
            raise RuntimeError(f"无法打开灵巧手端口 {port}")

        if self.portHandler.setBaudRate(baudrate):
            print(f"灵巧手波特率设置为 {baudrate}")
        else:
            raise RuntimeError("灵巧手波特率设置失败")

    def execute_action(self, id_list):
        for servo_id in id_list:
            scs_comm_result, scs_error = self.packetHandler.WritePos(servo_id, self.rightHand[servo_id], 0, 1500)
            if scs_comm_result != COMM_SUCCESS:
                print(f"通信错误 ID {servo_id}: {self.packetHandler.getTxRxResult(scs_comm_result)}")
            if scs_error != 0:
                print(f"舵机错误 ID {servo_id}: {self.packetHandler.getRxPacketError(scs_error)}")
            time.sleep(0.4)
        print("灵巧手动作完成")

    def stretch_hand(self):
        self.rightHand[2] = 423
        self.rightHand[4] = 596
        self.rightHand[6] = 329
        self.rightHand[8] = 810
        self.rightHand[9] = 339
        self.rightHand[10] = 633
        self.rightHand[11] = 495
        print("执行灵巧手伸直动作...")
        self.execute_action([10, 9, 8, 6, 4, 2, 11])

    def grasp_hand(self):
        self.rightHand[9] = 268
        self.rightHand[10] = 1010
        self.rightHand[8] = 203
        self.rightHand[6] = 995
        self.rightHand[4] = 82
        self.rightHand[2] = 1018
        print("执行灵巧手抓握动作...")
        self.execute_action([9, 2, 10, 4, 6, 8])

    def close(self):
        self.portHandler.closePort()


driver = SCServoDriver(port="COM8", baudrate=115200)
hand_controller = HandController()


class OrangeSpatialController(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.sendProcessingToPipeline(True)

        self.target_labels = ["orange", "sports ball"]
        self.min_conf_by_label = {
            "orange": float(args.orangeMinConf),
            "sports ball": float(args.ballMinConf),
        }
        self.hold_seconds = max(0.0, float(args.holdMs) / 1000.0)
        self.last_seen = {}

        self.motor_driver = driver
        self.hand_controller = hand_controller
        self.arm_sequence = [(0x4, 115), (0x12, 75), (0x11, 75), (0x4, 120), (0x11, 65)]
        self.initial_arm = {0x4: 85, 0x11: 105, 0x8: 95}

        self.motor_time_ms = 1100
        self.motor_speed = 1000

        self.motor8_far_x = -205.0
        self.motor8_near_x = -55.0
        self.motor8_far_angle = 95.0
        self.motor8_near_angle = 70.0
        self.last_motor8_angle = None
        self.motor_update_interval = 0.2
        self.last_motor_update = 0.0
        self.angle_threshold = 0.6

        self.reference_avg_x = None
        self.avg_stable_start = None
        self.avg_stable_duration = 3.0
        self.avg_stable_tolerance = 5.0
        self.grasp_in_progress = False
        self.has_grasped = False

        self.current_status = "Idle"
        self.orange_x_history = deque(maxlen=10)

        for servo_id in {0x4, 0x8, 0x11, 0x12}:
            self.motor_driver.set_torque_enable(servo_id, 1)
            time.sleep(0.05)

    def build(self, depth: dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb)

    def process(self, depthPreview, detections, rgbPreview):
        rgbPreview = rgbPreview.getCvFrame()
        now = time.time()

        target_dets = [d for d in detections.detections if d.labelName in self.target_labels]

        filtered_detections = []
        below_threshold = []
        for det in target_dets:
            min_conf = self.min_conf_by_label.get(det.labelName, 0.25)
            if det.confidence >= min_conf:
                filtered_detections.append(det)
            else:
                below_threshold.append(det)

        best_by_label = {}
        for det in filtered_detections:
            prev = best_by_label.get(det.labelName)
            if prev is None or det.confidence > prev.confidence:
                best_by_label[det.labelName] = det

        for label, det in best_by_label.items():
            self.last_seen[label] = {
                "label": label,
                "ts": now,
                "xmin": float(det.xmin),
                "xmax": float(det.xmax),
                "ymin": float(det.ymin),
                "ymax": float(det.ymax),
                "conf": float(det.confidence),
                "x": int(det.spatialCoordinates.x),
                "y": int(det.spatialCoordinates.y),
                "z": int(det.spatialCoordinates.z),
            }

        held_items = []
        if self.hold_seconds > 0:
            active_labels = set(best_by_label.keys())
            for label in self.target_labels:
                if label in active_labels:
                    continue
                last = self.last_seen.get(label)
                if last and now - last["ts"] <= self.hold_seconds:
                    held_items.append(last)

        target_item = best_by_label.get("orange") or best_by_label.get("sports ball")
        self.update_motion(target_item)

        self.displayResults(rgbPreview, filtered_detections, held_items, below_threshold)

    def update_motion(self, target_detection):
        if target_detection is None:
            self.orange_x_history.clear()
            self.reference_avg_x = None
            self.avg_stable_start = None
            if not self.has_grasped:
                self.current_status = "Searching..."
            return

        x_mm = float(target_detection.spatialCoordinates.x)
        self.orange_x_history.append(x_mm)
        avg_x = sum(self.orange_x_history) / len(self.orange_x_history)
        smoothing_x = avg_x if len(self.orange_x_history) == self.orange_x_history.maxlen else x_mm
        label_name = target_detection.labelName
        self.current_status = f"{label_name} X(avg10): {smoothing_x:.1f} mm"

        now = time.time()
        if now - self.last_motor_update >= self.motor_update_interval and not self.grasp_in_progress:
            target_angle = self.map_x_to_motor8(smoothing_x)
            if self.last_motor8_angle is None or abs(target_angle - self.last_motor8_angle) >= self.angle_threshold:
                self.send_motor_angle(0x8, target_angle)
                self.last_motor8_angle = target_angle
                self.last_motor_update = now
                self.current_status = f"Aligning 0x8 -> {target_angle:.1f}°"

        if len(self.orange_x_history) == self.orange_x_history.maxlen:
            if self.reference_avg_x is None:
                self.reference_avg_x = smoothing_x
                self.avg_stable_start = now
            elif abs(smoothing_x - self.reference_avg_x) <= self.avg_stable_tolerance:
                if self.avg_stable_start is None:
                    self.avg_stable_start = now
                elif (
                    not self.grasp_in_progress
                    and not self.has_grasped
                    and (now - self.avg_stable_start) >= self.avg_stable_duration
                ):
                    self.start_grasp_sequence()
            else:
                self.reference_avg_x = smoothing_x
                self.avg_stable_start = now
        else:
            self.reference_avg_x = None
            self.avg_stable_start = None

    def map_x_to_motor8(self, x_mm: float) -> float:
        x_mm = min(max(x_mm, self.motor8_far_x), self.motor8_near_x)
        span = self.motor8_near_x - self.motor8_far_x
        if span == 0:
            return self.motor8_near_angle
        ratio = (x_mm - self.motor8_far_x) / span
        angle = self.motor8_far_angle + ratio * (self.motor8_near_angle - self.motor8_far_angle)
        return max(min(angle, max(self.motor8_far_angle, self.motor8_near_angle)), min(self.motor8_far_angle, self.motor8_near_angle))

    def send_motor_angle(self, servo_id: int, angle_deg: float):
        position = deg_to_position(angle_deg)
        self.motor_driver.set_position(servo_id=servo_id, position=position, time_ms=self.motor_time_ms, speed=self.motor_speed)

    def start_grasp_sequence(self):
        print("橙子已对准，开始执行机械臂+灵巧手抓取流程...")
        self.grasp_in_progress = True
        self.current_status = "Running grasp sequence"

        for servo_id, angle in self.arm_sequence:
            print(f"机械臂指令: ID 0x{servo_id:X} -> {angle}°")
            self.send_motor_angle(servo_id, angle)
            time.sleep(0.3)

        time.sleep(0.8)
        self.hand_controller.grasp_hand()

        # 抓取完成后，按要求将机械臂关节调整到固定姿态
        print("抓取完成，执行终止关节姿态...")
        self.send_motor_angle(0x4, 100)
        time.sleep(0.2)
        self.send_motor_angle(0x11, 100)
        time.sleep(0.2)
        self.send_motor_angle(0x12, 200)

        self.has_grasped = True
        self.grasp_in_progress = False
        self.reference_avg_x = None
        self.avg_stable_start = None
        self.orange_x_history.clear()
        self.current_status = "Grasp complete"

    def reset_arm(self):
        print("机械臂回到初始位置...")
        for servo_id, angle in self.initial_arm.items():
            self.send_motor_angle(servo_id, angle)
            time.sleep(0.2)
        self.last_motor8_angle = None
        self.orange_x_history.clear()
        self.reference_avg_x = None
        self.avg_stable_start = None

    def displayResults(self, rgbFrame, detections, held_items, below_threshold):
        height, width, _ = rgbFrame.shape

        orange_conf = 0.0
        ball_conf = 0.0
        for det in detections:
            if det.labelName == "orange":
                orange_conf = max(orange_conf, float(det.confidence))
            elif det.labelName == "sports ball":
                ball_conf = max(ball_conf, float(det.confidence))
                
        for det in below_threshold:
            if det.labelName == "orange":
                orange_conf = max(orange_conf, float(det.confidence))
            elif det.labelName == "sports ball":
                ball_conf = max(ball_conf, float(det.confidence))
                
        # Draw status
        cv2.putText(rgbFrame, self.current_status, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(
            rgbFrame,
            f"orange conf: {orange_conf:.2f} ball: {ball_conf:.2f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            rgbFrame,
            f"orange conf(max): {orange_conf:.2f} (min={self.min_conf_by_label.get('orange', 0.0):.2f})",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(rgbFrame, f"Status: {self.current_status}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if self.orange_x_history:
            avg_view = sum(self.orange_x_history) / len(self.orange_x_history)
            cv2.putText(rgbFrame, f"Avg10 X: {avg_view:.1f} mm", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
        if self.avg_stable_start:
            elapsed = time.time() - self.avg_stable_start
            cv2.putText(rgbFrame, f"Stable avg: {elapsed:.1f}/3.0s", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.putText(rgbFrame, "Press S: reset arm+hand", (10, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        for det in detections:
            self.drawDetections(rgbFrame, det, width, height, (255, 255, 255))

        for held in held_items:
            self.drawHeld(rgbFrame, held, width, height)

        if args.showBelowThreshold:
            for det in below_threshold:
                self.drawDetections(rgbFrame, det, width, height, (0, 0, 255))

        cv2.imshow("rgb", rgbFrame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.stopPipeline()
        elif key == ord("s"):
            print("按下 S，伸直灵巧手并复位机械臂...")
            self.hand_controller.stretch_hand()
            self.reset_arm()
            self.has_grasped = False
            self.grasp_in_progress = False
            self.current_status = "Reset complete"

    def drawDetections(self, frame, detection, frameWidth, frameHeight, color):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"{detection.confidence * 100:.2f}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X:{int(detection.spatialCoordinates.x)}mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y:{int(detection.spatialCoordinates.y)}mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z:{int(detection.spatialCoordinates.z)}mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    def drawHeld(self, frame, item, frameWidth, frameHeight):
        x1 = int(item["xmin"] * frameWidth)
        x2 = int(item["xmax"] * frameWidth)
        y1 = int(item["ymin"] * frameHeight)
        y2 = int(item["ymax"] * frameHeight)
        color = (160, 160, 160)
        cv2.putText(
            frame,
            f"{item.get('label', 'held')} (hold) {item['conf'] * 100:.1f}%",
            (x1 + 10, y1 + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


try:
    print("启动橙子抓取视觉控制...")
    with dai.Pipeline() as p:
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
        else:
            depthSource = p.create(dai.node.NeuralDepth).build(
                monoLeft.requestFullResolutionOutput(),
                monoRight.requestFullResolutionOutput(),
                dai.DeviceModelZoo.NEURAL_DEPTH_LARGE,
            )

        spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
            camRgb, depthSource, modelDescription
        )
        visualizer = p.create(OrangeSpatialController)

        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        spatialDetectionNetwork.setConfidenceThreshold(0.3)

        visualizer.build(
            spatialDetectionNetwork.passthroughDepth,
            spatialDetectionNetwork.out,
            spatialDetectionNetwork.passthrough,
        )

        print(f"Starting pipeline with depth source: {args.depthSource}")
        p.run()

except Exception as exc:
    print(f"发生错误: {exc}")
finally:
    try:
        driver.close()
    except Exception:
        pass
    try:
        hand_controller.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("系统已关闭")