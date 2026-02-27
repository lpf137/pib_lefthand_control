#!/usr/bin/env python3

import argparse
import os
import sys
import time
import threading
import queue
from collections import deque

import cv2
import depthai as dai

from arm_scservo_driver import SCServoDriver

sys.path.append(os.path.dirname(__file__))
from scservo_sdk import PortHandler, scscl, COMM_SUCCESS  # type: ignore

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

# 灵巧手（COM7）
hand_port = PortHandler("COM7")
hand_packet = scscl(hand_port)


def _open_hand_port_or_raise():
    if hand_port.openPort():
        print("成功打开端口 COM7")
    else:
        raise RuntimeError("打开端口失败")

    if hand_port.setBaudRate(115200):
        print("成功设置波特率 115200")
    else:
        raise RuntimeError("设置波特率失败")


def _hand_write_pos(servo_id: int, pulse: int, time_ms: int = 1500):
    scs_comm_result, scs_error = hand_packet.WritePos(servo_id, pulse, 0, time_ms)
    if scs_comm_result != COMM_SUCCESS:
        print(f"通信错误 ID {servo_id}: {hand_packet.getTxRxResult(scs_comm_result)}")
    if scs_error != 0:
        print(f"舵机错误 ID {servo_id}: {hand_packet.getRxPacketError(scs_error)}")


def hand_grasp(time_ms: int = 1500, delay_s: float = 0.2):
    """抓取动作（按你给的 grasp_hand 配置）"""
    seq = [
        (11, 350),
        (10, 50),
        (9, 576),
        (8, 1000),
        (6, 1001),
        (4, 1000),
        (2, 1000),
    ]
    print("执行抓握动作...")
    for sid, pulse in seq:
        _hand_write_pos(sid, pulse, time_ms=time_ms)
        time.sleep(delay_s)
    print("抓握完成")


def hand_stretch(time_ms: int = 1500, delay_s: float = 0.2):
    """伸直动作（按你给的 1~10 配置）"""
    seq = [
        (1, 491),
        (2, 460),
        (3, 474),
        (4, 234),
        (5, 491),
        (6, 143),
        (7, 474),
        (8, 390),
        (9, 271),
        (10, 511),
        (11, 406),
    ]
    print("执行伸直动作...")
    for sid, pulse in sorted(seq, key=lambda x: x[0], reverse=True):
        _hand_write_pos(sid, pulse, time_ms=time_ms)
        time.sleep(delay_s)
    print("伸直完成")


def _deg_to_pos(angle_deg: float) -> int:
    return int((angle_deg / 200.0) * 4095)


def arm_move_angle(servo_id: int, angle_deg: float, time_ms: int = 800, speed: int = 1000):
    driver.set_torque_enable(servo_id, 1)
    time.sleep(0.03)
    driver.set_position(servo_id=servo_id, position=_deg_to_pos(angle_deg), time_ms=time_ms, speed=speed)


def arm_move_sequence(commands, time_ms: int = 1000, speed: int = 1000, step_delay_s: float = 0.08):
    for servo_id, angle_deg in commands:
        arm_move_angle(servo_id, float(angle_deg), time_ms=time_ms, speed=speed)
        time.sleep(step_delay_s)


def arm_move_sequence_blocking(commands, time_ms: int = 1000, speed: int = 1000, settle_s: float = 1.0):
    """按顺序移动，并等待每个舵机动作完成。如果误差超过70(raw)则重试直到满足要求。"""
    max_error = 200  # 允许的最大误差（原始脉冲单位）

    for servo_id, angle_deg in commands:
        target_pos = _deg_to_pos(float(angle_deg))
        
        while True:
            # 发送指令
            arm_move_angle(servo_id, float(angle_deg), time_ms=time_ms, speed=speed)
            # 等待运动完成
            time.sleep((time_ms / 1000.0) + settle_s)

            # 验证位置
            pos = driver.read_position(servo_id)
            if pos is not None:
                 curr_angle = (pos / 4095.0) * 200.0
                 diff = abs(pos - target_pos)
                 print(f"  -> ID 0x{servo_id:X} 目标:{angle_deg:.1f}°({target_pos}) 实测:{curr_angle:.1f}°({pos}) 误差:{diff}")
                 
                 if diff <= max_error:
                     break # 误差合格，完成当前舵机
                 else:
                     print(f"     [!] 误差 {diff} > {max_error}，重新执行 ID 0x{servo_id:X} ...")
            else:
                 print(f"  -> ID 0x{servo_id:X} 读取位置失败，重试...")


def do_startup_reset():
    """程序启动时执行一次复位：灵巧手伸直 + 机械臂回到原来位置。"""
    print("\n=== 启动复位：灵巧手伸直 + 机械臂回零 ===")

    # 灵巧手伸直
    hand_stretch(time_ms=1500, delay_s=0.2)

    # 机械臂回到原来位置（按用户给定顺序）
    home = [
        (0x10, 100),
        (0x9, 90),
        (0x6, 100),
        (0x3, 90),
        (0x2, 190),
        (0x11, 110),
        (0x8, 100),
        (0x7, 0),
        (0x4, 100),
    ]
    arm_move_sequence_blocking(home, time_ms=1000, speed=1000, settle_s=0.5)
    print("=== 启动复位完成，开始视觉跟随 ===\n")


class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.sendProcessingToPipeline(True)

        # 视觉检测逻辑引用 vision.py
        self.target_labels = ["bottle", "orange"]
        self.min_conf_by_label = {
            "bottle": float(args.bottleMinConf),
            "orange": float(args.orangeMinConf),
        }
        self.hold_seconds = max(0.0, float(args.holdMs) / 1000.0)
        self.last_seen = {}

        # 舵机映射: x=-40mm -> 60°, x=49mm -> 85°
        self.motor_id = 0x6
        self.x_min = -48.0
        self.x_max = 77.0
        self.angle_at_x_min = 60.0
        self.angle_at_x_max = 85.0
        self.last_angle = None
        self.angle_threshold = 0.5
        self.last_send_time = 0.0
        self.send_interval = 0.3
        self.target_time_ms = 800
        self.target_speed = 1000

        # 10帧平均值稳定2.5秒触发抓取
        self.x_buffer = deque(maxlen=10)
        self.stable_eps_mm = 2.0
        self.stable_required_s = 2.5
        self.stable_ref_x = None
        self.stable_start_ts = None

        # 状态
        self.state = "tracking"  # tracking -> grasping -> grasped -> resetting -> tracking
        self.last_key = -1
        
        # 任务队列和工作线程
        self.cmd_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        driver.set_torque_enable(self.motor_id, 1)
        time.sleep(0.2)

    def _worker_loop(self):
        """后台线程：处理耗时的机械臂/灵巧手动作，避免阻塞视频流"""
        print("工作线程已启动")
        while True:
            try:
                task = self.cmd_queue.get()
                action = task.get("action")
                
                if action == "grasp":
                    self._bg_grasp_sequence(task.get("angle"))
                elif action == "reset":
                    self._bg_stretch_reset()
                
                self.cmd_queue.task_done()
            except Exception as e:
                print(f"工作线程异常: {e}")

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

        control_item = best_by_label.get("bottle") or best_by_label.get("orange")
        if control_item is None and held_items:
            control_item = held_items[0]

        if control_item is not None:
            if self.state == "tracking":
                x_mm = self._extract_x(control_item)
                self.x_buffer.append(float(x_mm))
                avg_x = sum(self.x_buffer) / len(self.x_buffer)

                target_angle = self.map_x_to_angle(avg_x)
                cv2.putText(
                    rgbFrame,
                    f"x(avg10): {avg_x:.1f} mm",
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

                # 稳定检测（必须满10帧）
                if len(self.x_buffer) >= 10:
                    if self.stable_ref_x is None:
                        self.stable_ref_x = avg_x
                        self.stable_start_ts = now
                    else:
                        if abs(avg_x - self.stable_ref_x) <= self.stable_eps_mm:
                            stable_s = now - (self.stable_start_ts or now)
                            cv2.putText(
                                rgbFrame,
                                f"stable: {stable_s:.1f}/{self.stable_required_s:.1f}s",
                                (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            if stable_s >= self.stable_required_s:
                                self._do_grasp_sequence(current_angle_deg=target_angle)
                        else:
                            self.stable_ref_x = avg_x
                            self.stable_start_ts = now

            elif self.state == "grasped":
                cv2.putText(
                    rgbFrame,
                    "grasp done: press 's' to stretch + reset arm",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
            elif self.state in ["grasping", "resetting"]:
                 cv2.putText(
                    rgbFrame,
                    f"Action in progress... ({self.state})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        self.displayResults(rgbFrame, filtered_detections, held_items, below_threshold)

        if self.last_key == ord("s"):
            self._do_stretch_and_reset()

    def _do_grasp_sequence(self, current_angle_deg: float):
        if self.state != "tracking":
            return
        
        # 立即切换状态并放入队列，主线程不阻塞
        print("\n=== 触发抓取流程 (后台执行) ===")
        self.state = "grasping"
        self.cmd_queue.put({"action": "grasp", "angle": current_angle_deg})

    def _bg_grasp_sequence(self, current_angle_deg: float):
        # 实际的阻塞操作在后台线程运行
        # 防止异常回到 tracking 时立刻再次触发
        self.x_buffer.clear()
        self.stable_ref_x = None
        self.stable_start_ts = None

        try:
            # 机械臂动作：(0x10, 10), (0x6, 当前角度-7)
            # 0度可能触发限位保护导致不动，改为10度；settle_s=0.5减少等待
            target_angle_6 = float(current_angle_deg) - 2.0
            print(f"机械臂抓取动作(顺序执行): 0x10=0°->10°, 0x6={target_angle_6:.1f}°")
            arm_move_sequence_blocking(
                [(0x6, target_angle_6)],
                time_ms=1000,
                speed=1000,
                settle_s=0.5,
            )

            # 灵巧手抓握
            hand_grasp(time_ms=1500, delay_s=0.2)

            # 抓取完成后追加动作：0x9:90 0x10:0
            print("抓取完成后动作: 0x9=60°, 0x3=50°, 0x10=180°")
            arm_move_sequence_blocking(
                [(0x3, 50), (0x9, 60),(0x10, 180)],
                time_ms=500,
                speed=1000,
                settle_s=1.0,
            )

            print("=== 抓取流程完成，等待按 's' 伸直并复位 ===\n")
            self.state = "grasped"
        except Exception as e:
            print(f"抓取流程失败: {e}")
            self.state = "tracking"

    def _do_stretch_and_reset(self):
        print("\n=== 触发复位流程 (后台执行) ===")
        self.state = "resetting"
        self.cmd_queue.put({"action": "reset"})

    def _bg_stretch_reset(self):
        print("=== 执行伸直 + 机械臂复位 ===")
        try:
            hand_stretch(time_ms=1500, delay_s=0.2)


            home = [
                (0x10, 100),
                (0x9, 90),
                (0x6, 100),
                (0x3, 90),
                (0x2, 190),
                (0x12, 100),
                (0x11, 110),
                (0x8, 100),
                (0x7, 0),
                (0x4, 100),
            ]
            print("机械臂回到原来位置...")
            # 增加时间防止运动未完成，time_ms=2000, settle_s=1.0
            arm_move_sequence_blocking(home, time_ms=2000, speed=1000, settle_s=1.0)

            # 允许下一次抓取
            self.x_buffer.clear()
            self.stable_ref_x = None
            self.stable_start_ts = None
            self.last_angle = None
            self.last_send_time = 0.0
            self.state = "tracking"
            print("=== 复位完成，继续视觉跟随 ===\n")
        except Exception as e:
            print(f"伸直/复位失败: {e}")

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
        key = cv2.waitKey(1) & 0xFF
        self.last_key = key
        if key == ord("q"):
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
    print("映射关系: X=-40mm -> 60°, X=49mm -> 77°")

    _open_hand_port_or_raise()

    # 程序刚执行时，先进行一次复位
    do_startup_reset()

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
    try:
        driver.close()
    except Exception:
        pass
    try:
        hand_port.closePort()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("系统已关闭")