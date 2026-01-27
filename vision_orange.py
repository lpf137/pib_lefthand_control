#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

modelDescription = dai.NNModelDescription("yolov6-nano")
FPS = 30

class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb) # Must match the inputs to the process method

    def process(self, depthPreview, detections, rgbPreview):
        rgbPreview = rgbPreview.getCvFrame()
        self.displayResults(rgbPreview, detections.detections)

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
        if label not in ["orange","cup"]:  # 只显示水瓶
            return
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
    camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = p.create(dai.node.StereoDepth)
    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(camRgb, stereo, modelDescription, fps=FPS)
    visualizer = p.create(SpatialVisualizer)

    # setting node configs
    stereo.setExtendedDisparity(True)
    platform = p.getDefaultDevice().getPlatform()
    if platform == dai.Platform.RVC2:
        # For RVC2, width must be divisible by 16
        stereo.setOutputSize(640, 400)

    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)
    spatialDetectionNetwork.setConfidenceThreshold(0.3)  # 降低置信度阈值，提高识别率

    # Linking
    monoLeft.requestOutput((640, 400)).link(stereo.left)
    monoRight.requestOutput((640, 400)).link(stereo.right)

    visualizer.build(spatialDetectionNetwork.passthroughDepth, spatialDetectionNetwork.out, spatialDetectionNetwork.passthrough)

    p.run()