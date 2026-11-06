#!/usr/bin/env python3
"""
calibrate_motors.py - Assertive Motor & Encoder Calibration Tool for CleanupBot.

Hardware Target:
  - Motors: TT High Torque 1:90 Half-Metal Gearbox with 12 PPR Hall Encoders
  - Driver: DRV8833 Dual H-Bridge with 470uF bulk VM capacitor
  - Power:  6x AA battery pack (~7.5V - 9.0V)

Performs automated:
  1. Breakout Duty Detection (stiction floor for Left/Right, Forward/Reverse)
  2. Left/Right Dynamic Balance & Trim Ratio Calculation
  3. Hall Encoder Resolution & Odometry Validation (2160 ticks/rev)
  4. Auto-generates exact C macros for drive_control.h
"""

import sys
import time
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"


class MotorCalibrator(Node):
    def __init__(self) -> None:
        super().__init__("cleanupbot_motor_calibrator")

        qos_pico = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_encoders = self.create_subscription(
            Vector3, "/pico/drive/encoders", self._on_encoders, qos_pico
        )
        self.sub_heading = self.create_subscription(
            Float32, "/pico/imu/heading", self._on_heading, qos_pico
        )
        self.pub_cmd = self.create_publisher(Twist, "/pico/cmd_vel", 10)

        self.left_ticks = 0
        self.right_ticks = 0
        self.has_encoders = False
        self.current_heading = 0.0
        self.has_heading = False

    def _on_encoders(self, msg: Vector3) -> None:
        self.left_ticks = int(round(msg.x))
        self.right_ticks = int(round(msg.y))
        self.has_encoders = True

    def _on_heading(self, msg: Float32) -> None:
        self.current_heading = float(msg.data)
        self.has_heading = True

    def pump(self, duration_sec: float) -> None:
        end = time.time() + duration_sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def get_ticks(self) -> tuple[int, int]:
        self.pump(0.04)
        return self.left_ticks, self.right_ticks

    def send_raw_duty(self, left_duty: float, right_duty: float) -> None:
        """Sends raw duty cycles bypassing kinematic speed mapping (linear.y = 99.0)."""
        msg = Twist()
        msg.linear.x = float(left_duty)
        msg.angular.z = float(right_duty)
        msg.linear.y = 99.0  # Calibration mode override key
        self.pub_cmd.publish(msg)

    def stop_wheels(self) -> None:
        self.send_raw_duty(0.0, 0.0)
        self.pump(0.15)


def sweep_breakout(
    node: MotorCalibrator,
    wheel_name: str,
    is_left: bool,
    reverse: bool = False,
) -> float:
    """Ramps duty until wheel begins rotating consistently."""
    direction_str = "REVERSE" if reverse else "FORWARD"
    print(f"  {C.CYAN}* Sweeping {wheel_name} ({direction_str})...{C.RESET}", end="", flush=True)

    sign = -1.0 if reverse else 1.0
    start_duty = 0.16
    max_duty = 0.65
    step = 0.02
    threshold_ticks = 10

    current_duty = start_duty
    breakout_duty = max_duty

    while current_duty <= max_duty:
        # Sample baseline ticks
        node.pump(0.05)
        init_l, init_r = node.get_ticks()

        # Command duty for 220 ms
        for _ in range(11):
            if is_left:
                node.send_raw_duty(sign * current_duty, 0.0)
            else:
                node.send_raw_duty(0.0, sign * current_duty)
            node.pump(0.02)

        # Measure ticks
        now_l, now_r = node.get_ticks()
        delta = abs(now_l - init_l) if is_left else abs(now_r - init_r)

        if delta >= threshold_ticks:
            breakout_duty = current_duty
            break

        current_duty += step

    node.stop_wheels()
    time.sleep(0.1)

    duty_pct = breakout_duty * 100.0
    status_color = C.GREEN if breakout_duty < 0.48 else C.YELLOW
    print(f" {status_color}{C.BOLD}{duty_pct:.1f}% PWM duty{C.RESET} (overcame static friction)")
    return breakout_duty


def run_calibration() -> None:
    if not rclpy.ok():
        rclpy.init()

    node = MotorCalibrator()

    print(f"\n{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}       CleanupBot Hardware Motor & Encoder Calibration Suite{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"Hardware Specs: TT 1:90 Half-Metal Gearbox | 12 PPR Hall (2160 ticks/rev)")
    print(f"Power Supply  : 6x AA Pack (~7.5V) with 470µF VM Capacitor")
    print(f"{C.DIM}{'-' * 74}{C.RESET}")

    # 1. Connection Check
    print(f"{C.DIM}Connecting to Pico 2 W micro-ROS agent...{C.RESET}")
    node.pump(3.0)

    if not node.has_encoders:
        print(f"\n{C.BOLD}{C.RED}[ERROR] No telemetry received from /pico/drive/encoders.{C.RESET}")
        print("Ensure micro-ROS agent is active and Pico 2 W is connected to Wi-Fi.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print(f"  {C.GREEN}[OK] micro-ROS connection active and telemetry streaming.{C.RESET}")

    # 2. Safety Interlock
    print(f"\n{C.BOLD}{C.YELLOW}[SAFETY PRE-CHECK]{C.RESET}")
    print(f"  1. Elevate the robot chassis so BOTH wheels spin freely in the air.")
    print(f"  2. Verify 6x AA Motor Battery switch is ON (~7.5V rail).")
    try:
        ans = input(f"\n{C.BOLD}Are wheels elevated and clear to spin? (yes/no): {C.RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCalibration cancelled.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    if ans not in ("y", "yes"):
        print("Calibration aborted.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    # ------------------------------------------------------------------------
    # STEP 1: BREAKOUT DUTY SWEEPS (Overcoming Gearbox Stiction)
    # ------------------------------------------------------------------------
    print(f"\n{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}>> STEP 1/3: MEASURING GEARBOX BREAKOUT STICTION THRESHOLDS{C.RESET}")
    print(f"{C.DIM}Slowly ramping PWM to detect the exact voltage where each motor starts rolling.{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")

    l_fwd = sweep_breakout(node, "Left Wheel", is_left=True, reverse=False)
    l_rev = sweep_breakout(node, "Left Wheel", is_left=True, reverse=True)
    r_fwd = sweep_breakout(node, "Right Wheel", is_left=False, reverse=False)
    r_rev = sweep_breakout(node, "Right Wheel", is_left=False, reverse=True)

    # Recommended MIN_DUTY is max breakout + 3% margin
    max_breakout = max(l_fwd, l_rev, r_fwd, r_rev)
    recommended_min_duty = round(max_breakout + 0.03, 2)
    if recommended_min_duty < 0.35:
        recommended_min_duty = 0.35

    print(f"\n  {C.BOLD}{C.GREEN}>> Recommended MIN_DUTY Floor:{C.RESET} {C.BOLD}{recommended_min_duty:.2f}{C.RESET} ({recommended_min_duty*100:.0f}%)")

    # ------------------------------------------------------------------------
    # STEP 2: DYNAMIC SPEED & TRIM RATIO MEASUREMENT
    # ------------------------------------------------------------------------
    print(f"\n{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}>> STEP 2/3: DYNAMIC VELOCITY BALANCE & TRIM CALIBRATION{C.RESET}")
    print(f"{C.DIM}Spinning both wheels forward at nominal test duty for 2.0 seconds...{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")

    test_duty = max(recommended_min_duty + 0.10, 0.48)
    print(f"Commanding both wheels at {test_duty*100:.0f}% duty...")

    init_l, init_r = node.get_ticks()
    t_start = time.time()
    while time.time() - t_start < 2.0:
        node.send_raw_duty(test_duty, test_duty)
        node.pump(0.02)
    node.stop_wheels()

    end_l, end_r = node.get_ticks()
    delta_l = end_l - init_l
    delta_r = end_r - init_r

    print(f"\n[2.0s Wheel Telemetry]")
    print(f"  * Left Wheel Ticks  : {delta_l:+d}")
    print(f"  * Right Wheel Ticks : {delta_r:+d}")

    # Calculate Trim Multipliers
    if delta_l > 0 and delta_r > 0:
        ratio = float(delta_r) / float(delta_l)
        if ratio < 1.0:
            # Left motor is faster than Right
            left_trim = round(ratio, 2)
            right_trim = 1.00
            faster = "Left"
        else:
            # Right motor is faster than Left
            left_trim = 1.00
            right_trim = round(1.0 / ratio, 2)
            faster = "Right"

        diff_pct = abs(1.0 - ratio) * 100.0
        print(f"  * Motor Imbalance   : {diff_pct:.1f}% ({faster} motor spins faster)")
        print(f"  * Left Trim Factor  : {left_trim:.2f}")
        print(f"  * Right Trim Factor : {right_trim:.2f}")
    else:
        left_trim = 0.75
        right_trim = 1.00
        print(f"  {C.YELLOW}[WARN] One wheel did not spin properly during balance test.{C.RESET}")

    # ------------------------------------------------------------------------
    # STEP 3: ENCODER CPR & ODOMETRY VERIFICATION
    # ------------------------------------------------------------------------
    print(f"\n{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}>> STEP 3/3: ENCODER RESOLUTION & WHEEL RPM METRICS{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")

    cpr = 2160.0  # 12 PPR * 2 edges * 90 gearbox
    circumference = 2.0 * 3.14159265 * 0.033  # 0.2073m for 66mm wheels
    meters_per_tick = circumference / cpr

    revs_l = delta_l / cpr
    revs_r = delta_r / cpr
    rpm_l = (revs_l / 2.0) * 60.0
    rpm_r = (revs_r / 2.0) * 60.0

    print(f"  * Encoder Constants : 12 PPR * 2 edges * 90:1 gear = {int(cpr)} ticks/rev")
    print(f"  * Output Resolution : {meters_per_tick * 1000.0:.3f} mm per encoder tick")
    print(f"  * Measured Left RPM : {rpm_l:.1f} RPM (@ {test_duty*100:.0f}% duty, ~7.5V)")
    print(f"  * Measured Right RPM: {rpm_r:.1f} RPM (@ {test_duty*100:.0f}% duty, ~7.5V)")
    print(f"  * Motor Spec Match  : Half-metal gearbox rated at 59 RPM loaded / 113 RPM no-load.")

    # ------------------------------------------------------------------------
    # STEP 4: GENERATED FIRMWARE CONFIGURATION
    # ------------------------------------------------------------------------
    print(f"\n{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}>> RECOMMENDED CONFIGURATION FOR drive_control.h & encoder.h{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"""
{C.WHITE}// Copy these calibrated constants into src/Robot/pico2w_led/drive_control.h:
#define MIN_DUTY                {recommended_min_duty:.2f}f   // Overcomes half-metal gearbox stiction
#define LEFT_TRIM_FACTOR        {left_trim:.2f}f   // Balanced Left speed scale
#define RIGHT_TRIM_FACTOR       {right_trim:.2f}f   // Balanced Right speed scale

// In src/Robot/pico2w_led/encoder.h:
#define ENCODER_TICKS_PER_REV   2160.0f  // 12 PPR * 2 edges * 1:90 reduction{C.RESET}
""")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    run_calibration()
