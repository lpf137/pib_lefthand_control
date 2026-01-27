from decode_ir import NECDecoder

decoder = NECDecoder()

while True:
    pulses = capture_pulses(pin=你的GPIO)
    
    if pulses:
        result = decoder.decode(pulses)
        if result:
            address, command = result
            print("Address:", hex(address), "Command:", hex(command))
        else:
            print("Decode failed")
