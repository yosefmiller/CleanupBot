import threading
import time
from typing import Any

import rclpy
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


class PicoLedBridge:
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("rosa_pico_led_bridge")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._publisher = self._node.create_publisher(Bool, "/pico/led/set", qos)
        self._subscription = self._node.create_subscription(
            Bool,
            "/pico/led/state",
            self._on_state,
            qos,
        )
        self._call_lock = threading.Lock()
        self._state_received = False
        self._last_state = False

    def _on_state(self, message: Bool) -> None:
        self._last_state = message.data
        self._state_received = True

    def set_and_wait(self, state: bool, timeout_sec: float = 2.0) -> tuple[bool, str]:
        with self._call_lock:
            # 1. Verify that the Pico node subscriber is online
            discovery_deadline = time.monotonic() + timeout_sec
            while self._publisher.get_subscription_count() == 0:
                remaining = discovery_deadline - time.monotonic()
                if remaining <= 0:
                    return False, "Error: Pico LED is unreachable; no command subscriber discovered."
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            # 2. Publish state command
            self._state_received = False
            command = Bool()
            command.data = state
            self._publisher.publish(command)

            # 3. Wait for hardware loopback acknowledgment
            ack_deadline = time.monotonic() + timeout_sec
            while not self._state_received or self._last_state != state:
                remaining = ack_deadline - time.monotonic()
                if remaining <= 0:
                    return False, "Error: Pico command timed out without state acknowledgment."
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            return True, f"Success: Pico LED acknowledged state {'ON' if state else 'OFF'}."


_bridge: PicoLedBridge | None = None
_bridge_lock = threading.Lock()


def _get_bridge() -> PicoLedBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = PicoLedBridge()
        return _bridge


class SetPicoLedInput(BaseModel):
    """Input arguments for controlling the Raspberry Pi Pico 2 W onboard LED."""

    state: bool = Field(
        ...,
        description="A single flat boolean indicating the desired LED power state: True to turn the LED ON, False to turn the LED OFF. Do not nest inside an object.",
        examples=[True, False],
    )


@tool(args_schema=SetPicoLedInput)
def set_pico_led(state: bool) -> str:
    """Turn the Raspberry Pi Pico 2 W onboard LED on or off and verify acknowledgment."""
    _, message = _get_bridge().set_and_wait(state, timeout_sec=2.5)
    return message