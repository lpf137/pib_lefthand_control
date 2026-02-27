#!/usr/bin/env python3

import argparse
import time
import cv2
import depthai as dai
from arm_scservo_driver import SCServoDriver

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 30

parser = argparse.ArgumentParser()
parser.add_argument(
    "--depthSource", type=str, default="stereo", choices=["stereo", "neural"]
)
parser.add_argument(
    "--bottleMinConf", type=float, default=0.25, help="Minimum confidence for bottle"
)
parser.add_argument(
    "--orangeMinConf", type=float, default=0.12, help="Minimum confidence for orange"
)
parser.add_argument(
    "--ballMinConf", type=float, default=0.25, help="Minimum confidence for ball"
)
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

# 初始化电机驱动，仅控制ID 0x6
driver = SCServoDriver(port="COM8", baudrate=115200)


class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.sendProcessingToPipeline(True)

        # 视觉检测逻辑引用 vision.py
        self.target_labels = ["bottle", "orange", "sports ball"]
        self.min_conf_by_label = {
            "bottle": float(args.bottleMinConf),
            "orange": float(args.orangeMinConf),
            "sports ball": float(args.ballMinConf),
        }
        self.hold_seconds = max(0.0, float(args.holdMs) / 1000.0)
        self.last_seen = {}

        # 舵机映射: x=0mm -> 80°, x=120mm -> 100°
        self.motor_id = 0x6
        self.x_min = 0.0
        self.x_max = 120.0
        self.angle_at_x_min = 80.0
        self.angle_at_x_max = 100.0
        self.last_angle = None
        self.angle_threshold = 0.5
        self.last_send_time = 0.0
        self.send_interval = 0.3
        self.target_time_ms = 800
        self.target_speed = 1000

        driver.set_torque_enable(self.motor_id, 1)
        time.sleep(0.2)

    def build(self, depth: dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb)

    def _clamp(self, value, v_min, v_max):
        return max(v_min, min(v_max, value))

    def map_x_to_angle(self, x_mm: float) -> float:
        x_clamped = self._clamp(x_mm, self.x_min, self.x_max)
        span = self.x_max - self.x_min
        ratio = 0.0 if span == 0 else (x_clamped - self.x_min) / span
        angle = self.angle_at_x_min + ratio * (self.angle_at_x_max - self.angle_at_x_min)
        return angle

    def _extract_x(self, item) -> float:
        if isinstance(item, dict):
            return float(item.get("x", 0.0))
        return float(item.spatialCoordinates.x)

    def process(self, depthPreview, detections, rgbPreview):
        rgbFrame = rgbPreview.getCvFrame()
        now = time.time()

        target_dets = [d for d in detections.detections if d.labelName in self.target_labels]

        filtered_detections = []
        below_threshold = []
        for d in target_dets:
            min_conf = self.min_conf_by_label.get(d.labelName, 0.25)
            if d.confidence >= min_conf:
                filtered_detections.append(d)
            else:
                below_threshold.append(d)

        best_by_label = {}
        for d in filtered_detections:
            prev = best_by_label.get(d.labelName)
            if prev is None or d.confidence > prev.confidence:
                best_by_label[d.labelName] = d

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
            active = set(best_by_label.keys())
            for label in self.target_labels:
                if label in active:
                    continue
                last = self.last_seen.get(label)
                if last and now - last["ts"] <= self.hold_seconds:
                    held_items.append(last)

        control_item = best_by_label.get("bottle") or best_by_label.get("orange") or best_by_label.get("sports ball")
        if control_item is None and held_items:
            control_item = held_items[0]

        if control_item is not None:
            x_mm = self._extract_x(control_item)
            target_angle = self.map_x_to_angle(x_mm)
            cv2.putText(
                rgbFrame,
                f"control x: {x_mm:.1f} mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                rgbFrame,
                f"servo 0x6 -> {target_angle:.1f} deg",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if now - self.last_send_time >= self.send_interval:
                if self.last_angle is None or abs(target_angle - self.last_angle) >= self.angle_threshold:
                    target_pos = int((target_angle / 200.0) * 4095)
                    driver.set_position(
                        servo_id=self.motor_id,
                        position=target_pos,
                        time_ms=self.target_time_ms,
                        speed=self.target_speed,
                    )
                    self.last_angle = target_angle
                    self.last_send_time = now

        self.displayResults(rgbFrame, filtered_detections, held_items, below_threshold)

    def displayResults(self, rgbFrame, detections, held_items, below_threshold):
        height, width, _ = rgbFrame.shape

        orange_conf = 0.0
        for d in detections:
            if d.labelName == "orange":
                orange_conf = max(orange_conf, float(d.confidence))
        for d in below_threshold:
            if d.labelName == "orange":
                orange_conf = max(orange_conf, float(d.confidence))
        cv2.putText(
            rgbFrame,
            f"orange conf(max): {orange_conf:.2f} (min={self.min_conf_by_label.get('orange', 0.0):.2f})",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        for detection in detections:
            self.drawDetections(rgbFrame, detection, width, height)

        for item in held_items:
            self.drawHeld(rgbFrame, item, width, height)

        if args.showBelowThreshold:
            for detection in below_threshold:
                self.drawDetections(rgbFrame, detection, width, height, color=(0, 0, 255))

        cv2.imshow("rgb", rgbFrame)
        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()

    def drawDetections(self, frame, detection, frameWidth, frameHeight, color=(255, 255, 255)):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(
            frame,
            "{:.2f}".format(detection.confidence * 100),
            (x1 + 10, y1 + 35),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.putText(
            frame,
            f"X: {int(detection.spatialCoordinates.x)} mm",
            (x1 + 10, y1 + 50),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.putText(
            frame,
            f"Y: {int(detection.spatialCoordinates.y)} mm",
            (x1 + 10, y1 + 65),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.putText(
            frame,
            f"Z: {int(detection.spatialCoordinates.z)} mm",
            (x1 + 10, y1 + 80),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    def drawHeld(self, frame, item, frameWidth, frameHeight):
        x1 = int(item["xmin"] * frameWidth)
        x2 = int(item["xmax"] * frameWidth)
        y1 = int(item["ymin"] * frameHeight)
        y2 = int(item["ymax"] * frameHeight)
        color = (160, 160, 160)
        cv2.putText(
            frame,
            f"{item.get('label', 'held')} (hold) {item['conf']*100:.1f}%",
            (x1 + 10, y1 + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.putText(
            frame,
            f"X: {int(item['x'])} mm",
            (x1 + 10, y1 + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.putText(
            frame,
            f"Y: {int(item['y'])} mm",
            (x1 + 10, y1 + 55),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.putText(
            frame,
            f"Z: {int(item['z'])} mm",
            (x1 + 10, y1 + 70),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


try:
    print("开始视觉控制系统 (只控制电机 0x6)...")
    print("映射关系: X=0mm -> 80°, X=120mm -> 100°")

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

        print("Starting pipeline with depth source:", args.depthSource)
        p.run()

except Exception as e:
    print(f"发生错误: {e}")
finally:
    driver.close()
    cv2.destroyAllWindows()
    print("系统已关闭")