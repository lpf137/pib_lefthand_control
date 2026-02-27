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
    "--bottleMinConf", type=float, default=0.30, help="Minimum confidence for bottle"
)
parser.add_argument(
    "--orangeMinConf", type=float, default=0.12, help="Minimum confidence for orange"
)
parser.add_argument(
    "--ballMinConf", type=float, default=0.20, help="Minimum confidence for ball"
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

# 初始化电机驱动，仅控制ID 0x6 等
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


def hand_grasp_execute(seq, time_ms: int = 1500, delay_s: float = 0.2):
    """执行通用的灵巧手抓取序列"""
    print("执行灵巧手抓握动作...")
    for sid, pulse in seq:
        _hand_write_pos(sid, pulse, time_ms=time_ms)
        time.sleep(delay_s)
    print("抓握完成")


def hand_stretch(time_ms: int = 1500, delay_s: float = 0.2):
    """伸直动作（默认配置）"""
    seq = [
        (1, 491), (2, 460), (3, 474), (4, 234), (5, 491),
        (6, 143), (7, 474), (8, 390), (9, 271), (10, 511), (11, 406),
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


def arm_move_sequence_blocking(commands, time_ms: int = 1000, speed: int = 1000, settle_s: float = 1.0):
    """按顺序移动，并等待每个舵机动作完成。如果误差超过200(raw)则重试直到满足要求。"""
    max_error = 200  # 允许的最大误差（原始脉冲单位）
    max_retries = 5

    for servo_id, angle_deg in commands:
        target_pos = _deg_to_pos(float(angle_deg))
        retry_count = 0
        
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
                     retry_count += 1
                     if retry_count >= max_retries:
                         print(f"     [!] 警告: 达到最大重试次数 ({max_retries})，强制跳过 ID 0x{servo_id:X} 以防过热")
                         break
                     print(f"     [!] 误差 {diff} > {max_error}，第 {retry_count}/{max_retries} 次重试 ID 0x{servo_id:X} ...")
            else:
                 retry_count += 1
                 if retry_count >= max_retries:
                     print(f"     [!] 警告: 读取失败次数过多，强制跳过 ID 0x{servo_id:X}")
                     break
                 print(f"  -> ID 0x{servo_id:X} 读取位置失败，重试...")


def do_startup_reset():
    """程序启动时执行一次复位"""
    print("\n=== 启动复位：灵巧手伸直 + 机械臂回零 ===")
    hand_stretch(time_ms=1500, delay_s=0.2)
    home = [
        (0x10, 100), (0x9, 98), (0x6, 100), (0x3, 90),
        (0x2, 190), (0x12, 100), (0x11, 110), (0x8, 100),
        (0x7, 0), (0x4, 100),
    ]
    arm_move_sequence_blocking(home, time_ms=1000, speed=1000, settle_s=0.5)
    print("=== 启动复位完成，开始视觉跟随 ===\n")


# === 物品配置 ===
CONFIGS = {
    "orange": {
        "x_min": -40.0, "x_max": 49.0,
        "angle_min": 60.0, "angle_max": 77.0,
        "prep_angle_offset": -7.0, # Target = Current - 7
        # 橙子：arm prep 包括 ID 0x10 到 10度
        "arm_prep_func": lambda ang: [(0x10, 10), (0x6, ang)],
        "hand_seq": [
            (10, 50), (9, 350), (8, 1000), (6, 1001), (4, 1000), (2, 1000)
        ],
        "arm_post": [(0x3, 50), (0x9, 60), (0x10, 200)],
    },
    "bottle": {
        "x_min": -48.0, "x_max": 77.0,
        "angle_min": 60.0, "angle_max": 85.0,
        "prep_angle_offset": -2.0, # Target = Current - 2
        # 瓶子：arm prep 只有 ID 0x6
        "arm_prep_func": lambda ang: [(0x6, ang)],
        "hand_seq": [
             (11, 350), (10, 50), (9, 576), (8, 1000), (6, 1001), (4, 1000), (2, 1000)
        ],
        "arm_post": [(0x3, 50), (0x9, 60), (0x10, 180)],
    },
    "sports ball": {
        "x_min": -70.0, "x_max": 26.0,
        "angle_min": 60.0, "angle_max": 77.0,
        "prep_angle_offset": -5.0, # copy from orange
        "arm_prep_func": lambda ang: [(0x10, 0), (0x6, ang)],
        "hand_seq": [ # copy from orange
            (9, 290), (10,10), (8, 1000)
        ],
        "arm_post": [(0x3, 50), (0x9, 60)],
    }
}


class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.sendProcessingToPipeline(True)

        self.target_labels = ["bottle", "orange", "sports ball"]
        self.min_conf_by_label = {
            "bottle": float(args.bottleMinConf),
            "orange": float(args.orangeMinConf),
            "sports ball": float(args.ballMinConf),
        }
        self.hold_seconds = max(0.0, float(args.holdMs) / 1000.0)
        self.last_seen = {}

        self.motor_id = 0x6
        self.last_angle = None
        self.angle_threshold = 0.5
        self.last_send_time = 0.0
        self.send_interval = 0.3
        self.target_time_ms = 800
        self.target_speed = 1000

        # 稳定监测
        self.x_buffer = deque(maxlen=10)
        self.stable_eps_mm = 2.0
        self.stable_required_s = 2.5
        self.stable_ref_x = None
        self.stable_start_ts = None
        self.current_track_label = None # 当前锁定的目标物品类型

        self.state = "tracking"
        self.last_key = -1
        
        self.cmd_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        driver.set_torque_enable(self.motor_id, 1)
        time.sleep(0.2)

    def _worker_loop(self):
        print("工作线程已启动")
        while True:
            try:
                task = self.cmd_queue.get()
                action = task.get("action")
                
                if action == "grasp":
                    self._bg_grasp_sequence(task.get("angle"), task.get("label"))
                elif action == "reset":
                    self._bg_stretch_reset()
                
                self.cmd_queue.task_done()
            except Exception as e:
                print(f"工作线程异常: {e}")

    def build(self, depth: dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb)

    def _clamp(self, value, v_min, v_max):
        return max(v_min, min(v_max, value))

    def map_x_to_angle(self, x_mm: float, label: str) -> float:
        cfg = CONFIGS.get(label, CONFIGS["orange"])
        x_min, x_max = cfg["x_min"], cfg["x_max"]
        ang_min, ang_max = cfg["angle_min"], cfg["angle_max"]
        
        x_clamped = self._clamp(x_mm, x_min, x_max)
        span = x_max - x_min
        ratio = 0.0 if span == 0 else (x_clamped - x_min) / span
        angle = ang_min + ratio * (ang_max - ang_min)
        return angle

    def _extract_x(self, item) -> float:
        if isinstance(item, dict):
            return float(item.get("x", 0.0))
        return float(item.spatialCoordinates.x)

    def process(self, depthPreview, detections, rgbPreview):
        rgbFrame = rgbPreview.getCvFrame()
        now = time.time()
        
        # 全局按键检测
        key = cv2.waitKey(1) & 0xFF
        self.last_key = key
        if key == ord("q"):
            self.stopPipeline()
        if key == ord("s"):
            self._do_stretch_and_reset()

        # 如果不处于 tracking 状态，停止视觉识别逻辑和控制，仅显示画面
        if self.state != "tracking":
            msg = f"Action: {self.state}"
            if self.state == "grasped":
                msg = "Grasped. Press 's' to reset."
            cv2.putText(rgbFrame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("rgb", rgbFrame)
            return

        # ------------------------
        # 以下仅在 Tracking 状态执行
        # ------------------------

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
             # 同类别取置信度最高
            prev = best_by_label.get(d.labelName)
            if prev is None or d.confidence > prev.confidence:
                best_by_label[d.labelName] = d

        # 更新 last_seen
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

        # 决策逻辑：优先处理视野内（实时）置信度最高的目标
        control_item = None
        
        # 实时检测中，哪个置信度高就跟哪个
        bottle_det = best_by_label.get("bottle")
        orange_det = best_by_label.get("orange")
        ball_det = best_by_label.get("sports ball")
        
        candidates = [d for d in [bottle_det, orange_det, ball_det] if d is not None]
        if candidates:
            # 选置信度最高的
            control_item = max(candidates, key=lambda d: d.confidence)
        
        # 如果实时没有，找暂存的
        if control_item is None and held_items:
            # 简单去第一个
            control_item = held_items[0]

        if control_item is not None:
            # 只有 Tracking 状态下才计算和跟随
            if self.state == "tracking":
                
                label = control_item.labelName if hasattr(control_item, "labelName") else control_item.get("label")
                
                # 若目标切换，重置稳定器
                if label != self.current_track_label:
                    self.x_buffer.clear()
                    self.stable_ref_x = None
                    self.stable_start_ts = None
                    self.current_track_label = label

                x_mm = self._extract_x(control_item)
                self.x_buffer.append(float(x_mm))
                avg_x = sum(self.x_buffer) / len(self.x_buffer)

                target_angle = self.map_x_to_angle(avg_x, label)
                
                cv2.putText(rgbFrame, f"Target: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(rgbFrame, f"x(avg10): {avg_x:.1f} mm", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(rgbFrame, f"servo 0x6 -> {target_angle:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 舵机跟随
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

                # 稳定检测
                if len(self.x_buffer) >= 10:
                    if self.stable_ref_x is None:
                        self.stable_ref_x = avg_x
                        self.stable_start_ts = now
                    else:
                        if abs(avg_x - self.stable_ref_x) <= self.stable_eps_mm:
                            stable_s = now - (self.stable_start_ts or now)
                            cv2.putText(
                                rgbFrame, f"stable: {stable_s:.1f}/{self.stable_required_s:.1f}s",
                                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                            )
                            if stable_s >= self.stable_required_s:
                                self._do_grasp_sequence(current_angle_deg=target_angle, label=label)
                        else:
                            self.stable_ref_x = avg_x
                            self.stable_start_ts = now

        self.displayResults(rgbFrame, filtered_detections, held_items, below_threshold)

    def _do_grasp_sequence(self, current_angle_deg: float, label: str):
        if self.state != "tracking":
            return
        print(f"\n=== 触发抓取流程: {label} (后台执行) ===")
        self.state = "grasping"
        self.cmd_queue.put({"action": "grasp", "angle": current_angle_deg, "label": label})

    def _bg_grasp_sequence(self, current_angle_deg: float, label: str):
        self.x_buffer.clear()
        self.stable_ref_x = None
        self.stable_start_ts = None
        
        # 获取对应配置
        cfg = CONFIGS.get(label)
        if not cfg:
            print(f"找不到 {label} 的抓取配置！")
            self.state = "tracking"
            return

        try:
            # 1. 机械臂准备动作
            target_angle_6 = float(current_angle_deg) + cfg["prep_angle_offset"]
            arm_cmds = cfg["arm_prep_func"](target_angle_6)
            
            print(f"[{label}] 机械臂抓取位移动: {arm_cmds}")
            arm_move_sequence_blocking(
                arm_cmds,
                time_ms=1000,
                speed=1000,
                settle_s=0.5, # 统一用较快的 settle
            )

            # 2. 灵巧手抓握
            hand_seq = cfg["hand_seq"]
            hand_grasp_execute(hand_seq, time_ms=3000, delay_s=1.0)

            # 3. 抓取后动作
            post_cmds = cfg["arm_post"]
            print(f"[{label}] 抓取后机械臂抬起: {post_cmds}")
            arm_move_sequence_blocking(
                post_cmds,
                time_ms=700, # 700ms
                speed=1000,
                settle_s=1.0,
            )

            print("=== 抓取流程完成，等待按 's' 复位 ===\n")
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
            do_startup_reset() # 复用启动复位逻辑
            # 重置状态
            self.x_buffer.clear()
            self.stable_ref_x = None
            self.stable_start_ts = None
            self.last_angle = None
            self.last_send_time = 0.0
            self.current_track_label = None 
            self.state = "tracking"
            print("=== 复位完成，继续视觉跟随 ===\n")
        except Exception as e:
            print(f"伸直/复位失败: {e}")

    def displayResults(self, rgbFrame, detections, held_items, below_threshold):
        height, width, _ = rgbFrame.shape

        info_conf = {}
        # 收集最高置信度用于显示
        for label in self.target_labels:
            info_conf[label] = 0.0
            
        all_dets = list(detections) + below_threshold
        for d in all_dets:
            if d.labelName in info_conf:
                info_conf[d.labelName] = max(info_conf[d.labelName], float(d.confidence))
        
        info_str = " | ".join(f"{k}:{v:.2f}" for k, v in info_conf.items())
        
        cv2.putText(
            rgbFrame,
            f"Conf: {info_str}",
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
        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    def drawHeld(self, frame, item, frameWidth, frameHeight):
        x1 = int(item["xmin"] * frameWidth)
        x2 = int(item["xmax"] * frameWidth)
        y1 = int(item["ymin"] * frameHeight)
        y2 = int(item["ymax"] * frameHeight)
        color = (160, 160, 160)
        cv2.putText(frame, f"{item.get('label', 'held')} (hold)", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

try:
    print("开始视觉控制系统 (Orange & Bottle 合并版)...")
    _open_hand_port_or_raise()
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
                monoLeft.requestFullResolutionOutput(), monoRight.requestFullResolutionOutput(), dai.DeviceModelZoo.NEURAL_DEPTH_LARGE
            )

        spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(camRgb, depthSource, modelDescription)
        visualizer = p.create(SpatialVisualizer)

        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        visualizer.build(spatialDetectionNetwork.passthroughDepth, spatialDetectionNetwork.out, spatialDetectionNetwork.passthrough)
        print("Starting pipeline...")
        p.run()

except Exception as e:
    print(f"发生错误: {e}")
finally:
    try: driver.close()
    except: pass
    try: hand_port.closePort()
    except: pass
    cv2.destroyAllWindows()
