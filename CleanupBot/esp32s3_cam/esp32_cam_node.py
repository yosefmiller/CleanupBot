#!/usr/bin/env python3
"""ROS 2 Node publishing ESP32-S3 CAM frames to /camera/image_raw."""

import os
import sys
import threading
import time
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

# Set OpenCV FFmpeg network timeout to 5s (5,000,000 us) to fail fast instead of hanging 30s
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")


class ESP32CamNode(Node):

    def __init__(self, stream_url: str, flip_mode: int = None):
        super().__init__("esp32_cam_node")
        self.stream_url = stream_url

        self.declare_parameter("flip_mode", -99)
        param_flip = self.get_parameter("flip_mode").value
        if param_flip in (0, 1, -1):
            self.flip_mode = param_flip
        else:
            self.flip_mode = flip_mode

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.raw_pub = self.create_publisher(
            Image, "/camera/image_raw", camera_qos
        )
        self.compressed_pub = self.create_publisher(
            CompressedImage, "/camera/image_raw/compressed", camera_qos
        )
        self._running = True
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self.target_frame_period = 1.0 / 15.0

        self.get_logger().info(f"Connecting to ESP32-S3: {self.stream_url}")
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            self.get_logger().warn(
                f"Camera not reachable yet at {self.stream_url}. Background thread will auto-connect..."
            )
        else:
            self.get_logger().info("Camera stream opened successfully.")

        self._capture_thread = threading.Thread(target=self._drain_stream, daemon=True)
        self._capture_thread.start()

        self.timer = self.create_timer(
            self.target_frame_period, self.timer_callback
        )

    def _drain_stream(self):
        """Continuously grab latest frames. Auto-reconnects if stream drops or times out."""
        failed_reads = 0
        max_failed_reads = 10  # ~0.5s of consecutive drops before triggering reconnect

        while self._running:
            if not self.cap.isOpened():
                time.sleep(1.0)
                self.get_logger().info(f"Attempting connection to {self.stream_url}...")
                self.cap.open(self.stream_url)
                if self.cap.isOpened():
                    self.get_logger().info("Camera stream connected successfully.")
                    failed_reads = 0
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                failed_reads += 1
                time.sleep(0.05)
                # If stream timed out or stalled, release and force a clean reconnect
                if failed_reads >= max_failed_reads:
                    self.get_logger().warn(
                        "Camera stream stalled or disconnected. Reconnecting..."
                    )
                    self.cap.release()
                    with self._frame_lock:
                        self._latest_frame = None
                    time.sleep(1.0)
                continue

            failed_reads = 0
            with self._frame_lock:
                self._latest_frame = frame

    def timer_callback(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()

        # Optional host-side flip (0=vertical, 1=horizontal, -1=180 deg)
        if self.flip_mode is not None:
            frame = cv2.flip(frame, self.flip_mode)

        now = self.get_clock().now().to_msg()

        # Raw Image for Local RViz & YOLO Processing
        img_msg = Image()
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_optical_frame"
        img_msg.height, img_msg.width, _ = frame.shape
        img_msg.encoding = "bgr8"
        img_msg.step = 3 * img_msg.width
        img_msg.data = frame.tobytes()
        self.raw_pub.publish(img_msg)

        # JPEG encoding for remote subscribers / RViz
        if self.compressed_pub.get_subscription_count() > 0:
            comp_msg = CompressedImage()
            comp_msg.header.stamp = now
            comp_msg.header.frame_id = "camera_optical_frame"
            comp_msg.format = "jpeg"
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if encoded:
                comp_msg.data = buffer.tobytes()
                self.compressed_pub.publish(comp_msg)

    def destroy_node(self):
        self._running = False
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def parse_args():
    """Extract stream_url and optional --flip mode from command line."""
    stream_url = "http://127.0.0.1:8081/stream"
    flip_mode = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--flip" and i + 1 < len(args):
            try:
                flip_mode = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif not args[i].startswith("--") and not args[i].startswith("__"):
            stream_url = args[i]
            i += 1
        else:
            i += 1
    return stream_url, flip_mode


def main(args=None):
    rclpy.init(args=args)
    stream_url, flip_mode = parse_args()
    node = ESP32CamNode(stream_url, flip_mode=flip_mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
