from hx1838 import HX1838
import time

ir = HX1838()

while True:
    key = ir.ir_1838()
    if key != "":
        print("你按下了：", key)
    time.sleep(0.1)
