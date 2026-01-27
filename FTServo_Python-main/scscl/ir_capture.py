# ir_capture.py
import pigpio
import time

class IRCapture:
    def __init__(self, bcm_pin=18, gap_us=10000):
        """
        bcm_pin: 使用 BCM 编号（GPIO18 对应物理 12）
        gap_us: 认为一帧结束的空闲时间阈值（微秒）
        """
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Cannot connect to pigpiod. Start it with: sudo systemctl start pigpiod")
        self.pin = bcm_pin
        self.gap_us = gap_us
        self.pulses = []   # 存微秒值序列
        self.last_tick = None
        self.collecting = False
        # 注册回调，任何边沿都记录时间差（微秒）
        self.cb = self.pi.callback(self.pin, pigpio.EITHER_EDGE, self._cb)

    def _cb(self, gpio, level, tick):
        # tick 单位是微秒计数（wrap-around handled by pigpio.tickDiff）
        if self.last_tick is None:
            self.last_tick = tick
            return
        dt = pigpio.tickDiff(self.last_tick, tick)
        self.last_tick = tick
        self.pulses.append(dt)
        self.collecting = True

    def read_frame(self, timeout=1.0):
        """
        等待并返回一帧脉冲（list of ints, 单位 us），或者返回 None（超时/无数据）
        判定一帧的方式：在收集到脉冲后，如果出现 gap_us 的空闲时间，认为一帧结束。
        """
        start = time.time()
        while True:
            # 超时返回 None
            if time.time() - start > timeout:
                return None
            if self.collecting:
                # 如果最后一次时间已经大于 gap_us，认为帧结束
                # 我们通过检查 last recorded dt 是否大于 gap_us 来判断
                if len(self.pulses) > 0 and self.pulses[-1] > self.gap_us:
                    # 拷贝并清空缓冲
                    frame = self.pulses.copy()
                    self.pulses.clear()
                    self.collecting = False
                    # 去掉尾部最后那个长间隔（因为它只是分隔），以及可能的前导多余间隔
                    # 一般 frame 格式： [9000,4500,560,560,560,1690,..., >10000 ]
                    # 删除末尾的大 gap
                    while len(frame) and frame[-1] > self.gap_us:
                        frame.pop()
                    return frame
            time.sleep(0.001)

    def close(self):
        if self.cb:
            self.cb.cancel()
        if self.pi:
            self.pi.stop()
