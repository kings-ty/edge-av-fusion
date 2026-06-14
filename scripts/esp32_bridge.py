#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from av_fusion_interfaces.msg import AvDetection
import serial
import time

class Esp32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')
        # Jetson에서 ESP32는 보통 /dev/ttyUSB0 또는 /dev/ttyACM0로 잡힙니다.
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        except:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            
        self.subscription = self.create_subscription(
            AvDetection,
            '/av_fusion/detections',
            self.listener_callback,
            10)
        self.get_logger().info('ESP32 Serial Bridge Started')

    def listener_callback(self, msg):
        # fusion_state: 0(IDLE), 1(CANDIDATE), 2(TRIGGERED), 3(CONFIRMED)
        state_str = str(msg.fusion_state)
        self.ser.write(state_str.encode())
        
        if msg.fusion_state >= 2:
            self.get_logger().info(f'Sent Alert State {state_str} to ESP32')

def main(args=None):
    rclpy.init(args=args)
    bridge = Esp32Bridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.ser.close()
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
