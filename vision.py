#!/usr/bin/env python3

import argparse
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
        self.target_labels = ["bottle", "orange"]
        # 置信度阈值
        self.confidence_threshold = 0.3
        
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb) # Must match the inputs to the process method

    def process(self, depthPreview, detections, rgbPreview):
        rgbPreview = rgbPreview.getCvFrame()
        # 过滤检测结果
        filtered_detections = [
            d for d in detections.detections 
            if d.labelName in self.target_labels and d.confidence >= self.confidence_threshold
        ]
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
