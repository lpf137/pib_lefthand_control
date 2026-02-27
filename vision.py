#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import cv2
import depthai as dai
import numpy as np

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
    "--birdMinConf", type=float, default=0.5, help="Minimum confidence for bird"
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

if args.depthSource == "stereo":
    fps = STEREO_DEFAULT_FPS
else:
    fps = NEURAL_FPS

class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
        # 只识别这些标签
        self.target_labels = ["bottle", "orange", "bird", "sports ball"]
        # 不同目标使用不同置信度阈值（橙子通常更容易掉到低置信度）
        self.min_conf_by_label = {
            "bottle": float(args.bottleMinConf),
            "orange": float(args.orangeMinConf),
            "bird": float(args.birdMinConf), # 尝试用 bird 标签识别小黄鸭
            "sports ball": float(args.ballMinConf), # 识别纸球
        }

        # 检测抖动时短暂保持上一帧目标框，避免“标签经常不显示”
        self.hold_seconds = max(0.0, float(args.holdMs) / 1000.0)
        self.last_seen = {}  # label -> dict(ts,xmin,xmax,ymin,ymax,conf,x,y,z)
        
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb) # Must match the inputs to the process method

    def process(self, depthPreview, detections, rgbPreview):
        rgbPreview = rgbPreview.getCvFrame()
        now = time.time()

        target_dets = [d for d in detections.detections if d.labelName in self.target_labels]

        # 过滤检测结果（不同 label 使用不同阈值）
        filtered_detections = []
        below_threshold = []
        for d in target_dets:
            min_conf = self.min_conf_by_label.get(d.labelName, 0.25)
            if d.confidence >= min_conf:
                filtered_detections.append(d)
            else:
                below_threshold.append(d)

        # 更新 last_seen（每个标签保留当前置信度最高的一个）
        best_by_label = {}
        for d in filtered_detections:
            prev = best_by_label.get(d.labelName)
            if prev is None or d.confidence > prev.confidence:
                best_by_label[d.labelName] = d

        for label, d in best_by_label.items():
            self.last_seen[label] = {
                "label": label,
                "ts": now,
                "xmin": float(d.xmin),
                "xmax": float(d.xmax),
                "ymin": float(d.ymin),
                "ymax": float(d.ymax),
                "conf": float(d.confidence),
                "x": int(d.spatialCoordinates.x),
                "y": int(d.spatialCoordinates.y),
                "z": int(d.spatialCoordinates.z),
            }

        # 生成“保持”框：本帧没检测到，但在 hold 时间内出现过
        held_items = []
        if self.hold_seconds > 0:
            active_labels = set(best_by_label.keys())
            for label in self.target_labels:
                if label in active_labels:
                    continue
                last = self.last_seen.get(label)
                if not last:
                    continue
                if now - last["ts"] <= self.hold_seconds:
                    held_items.append(last)

        self.displayResults(rgbPreview, filtered_detections, held_items, below_threshold)

    def displayResults(self, rgbFrame, detections, held_items, below_threshold):
        height, width, _ = rgbFrame.shape

        # Debug overlay：显示橙子最高置信度（便于调阈值）
        orange_conf = 0.0
        for d in detections:
            if d.labelName == "orange":
                orange_conf = max(orange_conf, float(d.confidence))
        for d in below_threshold:
            if d.labelName == "orange":
                orange_conf = max(orange_conf, float(d.confidence))
        cv2.putText(
            rgbFrame,
            f"orange conf(max): {orange_conf:.2f}  (min={self.min_conf_by_label.get('orange', 0.0):.2f})",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        for detection in detections:
            self.drawDetections(rgbFrame, detection, width, height)

        # 画保持框（灰色）
        for item in held_items:
            self.drawHeld(rgbFrame, item, width, height)

        # 可选：画阈值以下（红色）用于调参
        if args.showBelowThreshold:
            for detection in below_threshold:
                self.drawDetections(rgbFrame, detection, width, height, color=(0, 0, 255))

        cv2.imshow("rgb", rgbFrame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def drawDetections(self, frame, detection, frameWidth, frameHeight, color=(255, 255, 255)):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
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
        cv2.putText(frame, f"X: {int(item['x'])} mm", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(item['y'])} mm", (x1 + 10, y1 + 55), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(item['z'])} mm", (x1 + 10, y1 + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

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
