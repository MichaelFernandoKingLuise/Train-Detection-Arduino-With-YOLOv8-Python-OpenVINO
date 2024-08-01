import time

def setup(board, servo_pins, ultrasonics):
    for trig, echo in ultrasonics:
        board.set_pin_mode_sonar(trig, echo)

    for pin in servo_pins:
        board.set_pin_mode_servo(pin)
        board.servo_write(pin, 0)
