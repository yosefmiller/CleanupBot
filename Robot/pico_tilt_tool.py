import threading
import time

import rclpy
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32

# Statuses the Pico sends in immediate reply to an arm request.
# Statuses 3-6 (running and stopped) are sent during physical movement.
ARM_REPLY_STATUSES = (1, 2, 7, 8)


class PicoTiltBridge:
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("rosa_pico_tilt_bridge")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._publisher = self._node.create_publisher(
            Int32, "/pico/tilt/request", qos
        )
        self._subscription = self._node.create_subscription(
            Int32, "/pico/tilt/status", self._on_status, qos
        )
        self._call_lock = threading.Lock()
        self._last_status: int | None = None
        self._status_count = 0

    def _on_status(self, message: Int32) -> None:
        self._last_status = message.data
        self._status_count += 1

    def _discard_old_statuses(self) -> None:
        # Process and discard stale status messages currently in the queue
        while True:
            count = self._status_count
            rclpy.spin_once(self._node, timeout_sec=0.05)
            if self._status_count == count:
                return

    def arm_and_wait(self, direction: str) -> str:
        direction = direction.strip().lower()
        if direction not in ("forward", "backward"):
            return "Invalid direction. Choose 'forward' or 'backward'."

        expected_status = 1 if direction == "forward" else 2
        command_value = 1 if direction == "forward" else -1

        with self._call_lock:
            # 1. Verify Pico subscription is online
            deadline = time.monotonic() + 5.0
            while self._publisher.get_subscription_count() == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "Error: Pico tilt controller is not connected; no request was sent."
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            # 2. Verify Pico status publisher is online
            deadline = time.monotonic() + 5.0
            while self._node.count_publishers("/pico/tilt/status") == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "Error: Pico tilt status is unavailable; no request was sent."
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            self._discard_old_statuses()
            self._last_status = None
            command = Int32()
            command.data = command_value
            self._publisher.publish(command)

            # 3. Wait for acknowledgment status from Pico
            deadline = time.monotonic() + 5.0
            while self._last_status not in ARM_REPLY_STATUSES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "Error: Pico did not acknowledge the tilt request; do not tilt."
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            if self._last_status == 8:
                return (
                    "The Pico is already armed or moving. Tilt the robot to complete "
                    "the current move or wait for it to finish, then try again."
                )
            if self._last_status == 7:
                return "The Pico reported an IMU sensor error; do not tilt."
            if self._last_status != expected_status:
                return (
                    f"Pico entered unexpected status {self._last_status}; do not tilt."
                )
            return f"Ready, please tilt {direction}."


_bridge: PicoTiltBridge | None = None
_bridge_lock = threading.Lock()


def _get_bridge() -> PicoTiltBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = PicoTiltBridge()
        return _bridge


class ArmTiltMoveInput(BaseModel):
    """Input arguments for arming tilt-activated robot motion."""

    direction: str = Field(
        ...,
        description="The direction to arm the robot: 'forward' or 'backward'.",
        examples=["forward", "backward"],
    )


@tool(args_schema=ArmTiltMoveInput, return_direct=True)
def arm_tilt_move(direction: str) -> str:
    """Arm one forward or backward tilt gesture on the Pico 2 W."""
    return _get_bridge().arm_and_wait(direction)
