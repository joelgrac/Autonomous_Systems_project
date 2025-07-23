#!/usr/bin/env python3
import rospy
import RPi.GPIO as GPIO
from geometry_msgs.msg import Twist

# === GPIO Setup ===
LEFT_FORWARD = 17
LEFT_BACKWARD = 27
RIGHT_FORWARD = 23
RIGHT_BACKWARD = 24
LEFT_PWM = 18
RIGHT_PWM = 25

FREQ = 100  # Hz
MAX_SPEED = 100  # PWM range (0–100)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [LEFT_FORWARD, LEFT_BACKWARD, RIGHT_FORWARD, RIGHT_BACKWARD, LEFT_PWM, RIGHT_PWM]:
    GPIO.setup(pin, GPIO.OUT)

pwm_left = GPIO.PWM(LEFT_PWM, FREQ)
pwm_right = GPIO.PWM(RIGHT_PWM, FREQ)
pwm_left.start(0)
pwm_right.start(0)

def stop_motors():
    GPIO.output(LEFT_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_BACKWARD, GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

def set_motor(forward, backward, speed, pwm):
    GPIO.output(forward, speed > 0)
    GPIO.output(backward, speed < 0)
    pwm.ChangeDutyCycle(min(abs(speed), MAX_SPEED))

def cmd_vel_callback(msg):
    linear = msg.linear.x
    angular = msg.angular.z

    left_speed = linear * 100 - angular * 50
    right_speed = linear * 100 + angular * 50

    rospy.loginfo(f"⚙️ cmd_vel: linear={linear:.2f}, angular={angular:.2f}")
    rospy.loginfo(f"➡️ left_speed={left_speed:.1f}, right_speed={right_speed:.1f}")

    set_motor(LEFT_FORWARD, LEFT_BACKWARD, left_speed, pwm_left)
    set_motor(RIGHT_FORWARD, RIGHT_BACKWARD, right_speed, pwm_right)

def motor_driver_node():
    rospy.init_node("motor_driver_node")
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    rospy.on_shutdown(stop_motors)
    rospy.loginfo("🚗 motor_driver_node ready")
    rospy.spin()

if __name__ == '__main__':
    try:
        motor_driver_node()
    except rospy.ROSInterruptException:
        stop_motors()
