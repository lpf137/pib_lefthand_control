import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)     # 使用物理引脚编号
GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Waiting for IR signal...")

while True:
    if GPIO.input(12) == 0:
        print("Pulse detected!")
    time.sleep(0.01)
