from ir_capture import IRCapture
from nec_decoder import NECDecoder

cap = IRCapture(bcm_pin=18)   # 捕获 GPIO18（物理12）
decoder = NECDecoder()

while True:
    frame = cap.read_frame(timeout=3.0)   # 返回一次完整按键的脉冲列表
    if frame:
        result = decoder.decode(frame)
        if result:
            addr, cmd = result
            print("Decoded: address=0x%02x, command=0x%02x" % (addr, cmd))
        else:
            print("Decode failed")
