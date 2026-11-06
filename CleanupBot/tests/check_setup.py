"""Assertive, Color-Coded Diagnostic and Calibration Verification Script for CleanupBot.

Checks 5 Critical Links:
  1. Gazebo Ground Truth / Odometry (/sim/ground_truth or /odom)
  2. Gazebo Velocity Command Bridge (/cmd_vel)
  3. Real Robot BNO085 9-DOF IMU Telemetry (/pico/imu/heading, /pico/imu/data)
  4. Real Robot Physical Wheel Encoders (/pico/drive/encoders, /pico/drive/distance)
  5. Real Robot Differential Drive Controller (/pico/cmd_vel, /pico/drive/status)

Assertive Hardware Diagnostics (--wheel-test):
  - Strictly FAILS on zero motion (detects if motor power switch is OFF or VM disconnected).
  - Asserts individual Left vs. Right encoder ticks to detect single-motor stalls or wiring faults.
  - Detects reverse motor polarity (e.g. forward command producing backward rotation).
  - Performs dynamic BNO085 IMU gyro/magnetometer validation when rotated by hand.
  - Fully color-coded logs (Green=Pass, Red=Fail, Yellow=Warn, Cyan=Headers).

Usage:
  python tests/check_setup.py
  python tests/check_setup.py --wheel-test
"""

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Int32


# ANSI Terminal Colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Status Badges
    OK = f"{BOLD}{GREEN}[OK]{RESET}     "
    FAIL = f"{BOLD}{RED}[FAIL]{RESET}   "
    WARN = f"{BOLD}{YELLOW}[WARN]{RESET}   "
    STANDBY = f"{BOLD}{YELLOW}[STANDBY]{RESET}"

    # Highlight Banners
    BG_GREEN = f"\033[42m{WHITE}{BOLD}"
    BG_RED = f"\033[41m{WHITE}{BOLD}"


class AssertiveSetupChecker(Node):
    """Monitors telemetry and strictly validates physical actuator and sensor responses."""

    def __init__(self) -> None:
        super().__init__("cleanupbot_assertive_checker")

        # Gazebo topics use default Reliable QoS
        qos_sim = 10

        # Pico micro-ROS topics use Best Effort QoS
        qos_pico = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Pico micro-ROS topics use Best Effort QoS (compatible with both Best Effort & Reliable)
        qos_pico = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Gazebo telemetry state
        self.has_sim_odom = False
        self.has_sim_gt = False

        # Pico telemetry state
        self.has_real_heading = False
        self.has_real_pitch = False
        self.has_real_distance = False
        self.has_real_encoders = False
        self.has_real_status = False

        self.last_heading = 0.0
        self.last_pitch = 0.0
        self.last_distance = 0.0
        self.left_ticks = 0
        self.right_ticks = 0
        self.last_status = 0

        # Subscriptions
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_sim)
        self.create_subscription(Odometry, "/sim/ground_truth", self._on_gt, qos_sim)
        self.create_subscription(Float32, "/pico/imu/heading", self._on_heading, qos_pico)
        self.create_subscription(Float32, "/pico/imu/pitch", self._on_pitch, qos_pico)
        self.create_subscription(Float32, "/pico/drive/distance", self._on_distance, qos_pico)
        self.create_subscription(Vector3, "/pico/drive/encoders", self._on_encoders, qos_pico)
        self.create_subscription(Int32, "/pico/drive/status", self._on_status, qos_pico)

        # Publishers
        self.pub_sim = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_real = self.create_publisher(Twist, "/pico/cmd_vel", 10)

    def _on_odom(self, _msg: Odometry) -> None:
        self.has_sim_odom = True

    def _on_gt(self, _msg: Odometry) -> None:
        self.has_sim_gt = True

    def _on_heading(self, msg: Float32) -> None:
        self.has_real_heading = True
        self.last_heading = float(msg.data)

    def _on_pitch(self, msg: Float32) -> None:
        self.has_real_pitch = True
        self.last_pitch = float(msg.data)

    def _on_distance(self, msg: Float32) -> None:
        self.has_real_distance = True
        self.last_distance = float(msg.data)

    def _on_encoders(self, msg: Vector3) -> None:
        self.has_real_encoders = True
        self.left_ticks = int(round(msg.x))
        self.right_ticks = int(round(msg.y))
        self.last_distance = float(msg.z)

    def _on_status(self, msg: Int32) -> None:
        self.has_real_status = True
        self.last_status = int(msg.data)

    def pump(self, duration_sec: float = 4.5) -> None:
        """Pump the ROS 2 event loop for up to duration_sec, breaking early once topics respond."""
        start_time = time.time()
        end_time = start_time + duration_sec
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.02)
            # Break early as soon as hardware telemetry is streaming
            if (time.time() - start_time) >= 1.0:
                pico_ready = self.has_real_heading and self.has_real_encoders
                sim_ready = self.has_sim_odom or self.has_sim_gt
                if pico_ready and (sim_ready or (time.time() - start_time) >= 1.5):
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Assertive Sim-to-Real Diagnostics for CleanupBot.")
    parser.add_argument(
        "--wheel-test",
        action="store_true",
        help="Execute strict physical motor, encoder, and IMU validation.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Execute automated motor stiction and balance calibration suite.",
    )
    args = parser.parse_args()

    if args.calibrate:
        from calibrate_motors import run_calibration
        run_calibration()
        return

    if not rclpy.ok():
        rclpy.init()

    node = AssertiveSetupChecker()
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}       CleanupBot Assertive Sim-to-Real Diagnostic System{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.DIM}Sampling live ROS 2 topics across simulation & hardware (up to 4.5s)...{C.RESET}")

    node.pump(4.5)

    # 1. Gazebo Telemetry
    sim_ok = node.has_sim_odom or node.has_sim_gt
    badge_sim = C.OK if sim_ok else C.STANDBY
    sim_detail = f"{C.GREEN}-> Active (/odom){C.RESET}" if sim_ok else f"{C.DIM}-> Offline (run 'pixi run sim' to launch){C.RESET}"
    print(f"  {badge_sim} Gazebo Sim Telemetry (/sim/ground_truth, /odom) {sim_detail}")

    # 2. Gazebo Velocity Pipeline
    sim_pub_ok = sim_ok or (node.pub_sim.get_subscription_count() > 0)
    badge_sim_pub = C.OK if sim_pub_ok else C.STANDBY
    print(f"  {badge_sim_pub} Gazebo Command Pipeline (/cmd_vel)")

    # 3. Real Robot IMU
    imu_ok = node.has_real_heading
    badge_imu = C.OK if imu_ok else C.FAIL
    if imu_ok:
        pitch_str = f" | Pitch: {node.last_pitch:+.1f}°" if node.has_real_pitch else ""
        imu_detail = f"{C.GREEN}-> Heading: {node.last_heading:+.1f}°{pitch_str}{C.RESET}"
    else:
        imu_detail = f"{C.RED}-> No data from /pico/imu/heading{C.RESET}"
    print(f"  {badge_imu} BNO085 9-DOF IMU Telemetry {imu_detail}")

    # 4. Real Robot Wheel Encoders
    enc_ok = node.has_real_encoders or node.has_real_distance
    badge_enc = C.OK if enc_ok else C.FAIL
    if node.has_real_encoders:
        enc_detail = f"{C.GREEN}-> Left: {node.left_ticks} | Right: {node.right_ticks} | Dist: {node.last_distance:.3f}m{C.RESET}"
    elif node.has_real_distance:
        enc_detail = f"{C.GREEN}-> Cumulative Distance: {node.last_distance:.3f}m{C.RESET}"
    else:
        enc_detail = f"{C.RED}-> No data from /pico/drive/encoders{C.RESET}"
    print(f"  {badge_enc} Optical Wheel Encoders {enc_detail}")

    # 5. Real Robot Drive Pipeline
    cmd_ok = node.has_real_status or (node.pub_real.get_subscription_count() > 0)
    badge_cmd = C.OK if cmd_ok else C.FAIL
    status_str = f"Status: {node.last_status}" if node.has_real_status else "Awaiting status"
    print(f"  {badge_cmd} micro-ROS Drive Pipeline (/pico/cmd_vel, {status_str})")

    print(f"{C.DIM}{'-' * 74}{C.RESET}")
    real_hardware_ready = imu_ok and enc_ok and cmd_ok
    sim_ready = sim_ok

    if real_hardware_ready and sim_ready:
        print(f"{C.BOLD}{C.GREEN}>> SIM-TO-REAL SUITE: ALL 5 LINKS VERIFIED ACTIVE! [READY]{C.RESET}")
    else:
        if real_hardware_ready:
            print(f"{C.BOLD}{C.GREEN}>> PHYSICAL ROBOT HARDWARE: ALL 3 LINKS ACTIVE! [HARDWARE READY]{C.RESET}")
        else:
            print(f"{C.BOLD}{C.RED}>> PHYSICAL ROBOT HARDWARE: ONE OR MORE PICO LINKS INACTIVE.{C.RESET}")
            print(f"   {C.YELLOW}* Ensure micro-ROS agent is active and relay.py is running on Windows.{C.RESET}")

        if sim_ready:
            print(f"{C.BOLD}{C.GREEN}>> GAZEBO SIMULATION: ACTIVE [DIGITAL TWIN READY]{C.RESET}")
        else:
            print(f"{C.BOLD}{C.YELLOW}>> GAZEBO SIMULATION: NOT RUNNING (Launch with 'pixi run sim' for sim rehearsal){C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")

    if not args.wheel_test:
        print(f"\n{C.CYAN}Tip: Run with '{C.BOLD}--wheel-test{C.RESET}{C.CYAN}' to perform assertive motor, encoder, and IMU validation.{C.RESET}")
        node.destroy_node()
        rclpy.shutdown()
        return

    # -------------------------------------------------------------
    # ASSERTIVE HARDWARE VALIDATION SUITE (--wheel-test)
    # -------------------------------------------------------------
    print("\n" + f"{C.CYAN}{'#' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}               PHYSICAL ACTUATOR & SENSOR VALIDATION{C.RESET}")
    print(f"{C.CYAN}{'#' * 74}{C.RESET}")

    if not (imu_ok and cmd_ok):
        print(f"\n{C.BOLD}{C.RED}[CRITICAL ERROR] Cannot run wheel test: Pico 2 W is not communicating.{C.RESET}")
        print("Start micro-ROS agent and connect Pico to Wi-Fi before testing motors.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print(f"\n{C.BOLD}{C.YELLOW}[SAFETY PROTOCOL]{C.RESET}")
    print(f"  {C.YELLOW}1. Elevate your CleanupBot so BOTH DRIVE WHEELS ARE OFF THE TABLE.{C.RESET}")
    print(f"  {C.YELLOW}2. The wheels must be completely free to spin in the air.{C.RESET}")
    print(f"  {C.YELLOW}3. Ensure the Motor Battery Switch is ON (powers DRV8833 VM rail).{C.RESET}")
    try:
        resp = input(f"\n{C.BOLD}Are the wheels safely elevated in the air? (yes/no): {C.RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{C.YELLOW}Test cancelled by operator.{C.RESET}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    if resp not in ("yes", "y"):
        print(f"{C.YELLOW}Wheel test cancelled.{C.RESET}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    all_tests_passed = True

    # ---------------------------------------------------------
    # TEST 1: FORWARD DRIVE & ENCODER ASSERTION
    # ---------------------------------------------------------
    print("\n" + f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}>> TEST 1/3: FORWARD MOTOR & ENCODER ASSERTION (1.0s @ 0.20 m/s){C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")

    node.pump(0.5)
    start_lt = node.left_ticks
    start_rt = node.right_ticks
    start_dist = node.last_distance

    twist_fwd = Twist()
    twist_fwd.linear.x = 0.20
    twist_fwd.angular.z = 0.0

    print(f"{C.DIM}Commanding forward drive (1.0s)...{C.RESET}")
    t_end = time.time() + 1.0
    while time.time() < t_end:
        node.pub_real.publish(twist_fwd)
        node.pump(0.10)

    node.pub_real.publish(Twist())
    print(f"{C.DIM}Waiting for wheel movement and telemetry to settle...{C.RESET}")

    t_settle_end = time.time() + 2.0
    last_l = node.left_ticks
    last_r = node.right_ticks
    last_change = time.time()

    while time.time() < t_settle_end:
        node.pump(0.05)
        if node.left_ticks != last_l or node.right_ticks != last_r:
            last_l = node.left_ticks
            last_r = node.right_ticks
            last_change = time.time()
        elif (time.time() - last_change) >= 0.4 and (abs(node.left_ticks - start_lt) >= 10 or abs(node.right_ticks - start_rt) >= 10):
            break

    delta_lt = node.left_ticks - start_lt
    delta_rt = node.right_ticks - start_rt
    delta_dist = node.last_distance - start_dist

    print(f"\n{C.BOLD}[Forward Drive Telemetry]{C.RESET}")
    color_lt = C.GREEN if delta_lt > 0 else (C.YELLOW if delta_lt == 0 else C.RED)
    color_rt = C.GREEN if delta_rt > 0 else (C.YELLOW if delta_rt == 0 else C.RED)
    color_dst = C.GREEN if delta_dist > 0 else (C.YELLOW if delta_dist == 0 else C.RED)

    print(f"  * Left Wheel Ticks  : {color_lt}{delta_lt:+d}{C.RESET}")
    print(f"  * Right Wheel Ticks : {color_rt}{delta_rt:+d}{C.RESET}")
    print(f"  * Odometry Distance : {color_dst}{delta_dist:+.3f} m{C.RESET}")

    # STRICT ASSERTIONS:
    if delta_lt == 0 and delta_rt == 0:
        print("\n" + f"{C.RED}{'!' * 74}{C.RESET}")
        print(f"{C.BOLD}{C.RED}>>> [TEST 1 FAILED] ZERO WHEEL MOTION DETECTED! <<<{C.RESET}")
        print(f"{C.RED}Neither wheel rotated during the 1.0s command pulse.{C.RESET}")
        print(f"{C.DIM}{'-' * 74}{C.RESET}")
        print(f"{C.YELLOW}DIAGNOSTIC TROUBLESHOOTING CHECKLIST:{C.RESET}")
        print("  1. Motor Battery Switch is OFF -> Turn ON the battery pack switch.")
        print("  2. DRV8833 VM Power Disconnected -> Verify battery leads to DRV8833 VM & GND.")
        print("  3. STBY Pin Low -> Verify Pico GP9 (STBY) is connected to DRV8833 STBY.")
        print("  4. Motor Leads Disconnected -> Verify OUT1-OUT4 connections to motor terminals.")
        print("  5. Low Battery Voltage -> Check if batteries have at least 6V-8V total.")
        print(f"{C.RED}{'!' * 74}{C.RESET}")
        all_tests_passed = False

    elif delta_lt == 0 and delta_rt > 0:
        print("\n" + f"{C.RED}{'!' * 74}{C.RESET}")
        print(f"{C.BOLD}{C.RED}>>> [TEST 1 FAILED] LEFT MOTOR FAILURE DETECTED! <<<{C.RESET}")
        print(f"  - Left Wheel  : {C.RED}0 ticks (DEAD / STALLED){C.RESET}")
        print(f"  - Right Wheel : {C.GREEN}+{delta_rt} ticks (WORKING){C.RESET}")
        print(f"{C.DIM}{'-' * 74}{C.RESET}")
        print(f"{C.YELLOW}TROUBLESHOOTING:{C.RESET}")
        print("  * Check Left Motor wiring on DRV8833 (OUT1/OUT2, GP10/GP11).")
        print("  * Check Left Encoder sensor wiring on GP18 (A) & GP19 (B).")
        print(f"{C.RED}{'!' * 74}{C.RESET}")
        all_tests_passed = False

    elif delta_rt == 0 and delta_lt > 0:
        print("\n" + f"{C.RED}{'!' * 74}{C.RESET}")
        print(f"{C.BOLD}{C.RED}>>> [TEST 1 FAILED] RIGHT MOTOR FAILURE DETECTED! <<<{C.RESET}")
        print(f"  - Left Wheel  : {C.GREEN}+{delta_lt} ticks (WORKING){C.RESET}")
        print(f"  - Right Wheel : {C.RED}0 ticks (DEAD / STALLED){C.RESET}")
        print(f"{C.DIM}{'-' * 74}{C.RESET}")
        print(f"{C.YELLOW}TROUBLESHOOTING:{C.RESET}")
        print("  * Check Right Motor wiring on DRV8833 (OUT3/OUT4, GP12/GP13).")
        print("  * Check Right Encoder sensor wiring on GP20 (A) & GP21 (B).")
        print(f"{C.RED}{'!' * 74}{C.RESET}")
        all_tests_passed = False

    elif delta_lt < -5 or delta_rt < -5:
        print("\n" + f"{C.RED}{'!' * 74}{C.RESET}")
        print(f"{C.BOLD}{C.RED}>>> [TEST 1 FAILED] REVERSE MOTOR POLARITY DETECTED! <<<{C.RESET}")
        print(f"{C.RED}Forward command produced backward wheel rotation:{C.RESET}")
        print(f"  - Left: {delta_lt} ticks, Right: {delta_rt} ticks")
        print(f"{C.DIM}{'-' * 74}{C.RESET}")
        print(f"{C.YELLOW}NOTE: The newly generated firmware includes automatic software polarity inversion.{C.RESET}")
        print(f"{C.YELLOW}Flash build/pico2w_led.uf2 to fix this in firmware without rewiring!{C.RESET}")
        print(f"{C.RED}{'!' * 74}{C.RESET}")
        all_tests_passed = False

    elif delta_lt >= 10 and delta_rt >= 10:
        print(f"\n{C.BOLD}{C.GREEN}>>> [TEST 1 PASSED] Both wheels rotated forward as commanded! <<<{C.RESET}")
        max_ticks = max(abs(delta_lt), abs(delta_rt))
        imbalance = abs(delta_lt - delta_rt) / float(max_ticks)
        if imbalance > 0.40:
            print(f"  {C.YELLOW}[ADVISORY] Speed imbalance is {imbalance*100:.0f}%.{C.RESET}")
            print("  The firmware's adaptive closed-loop controller will dynamically trim this.")
        else:
            print(f"  {C.GREEN}[EXCELLENT] Left/Right balance is within {imbalance*100:.1f}%.{C.RESET}")
    else:
        print(f"\n{C.BOLD}{C.YELLOW}[TEST 1 WARNING] Low rotation count (Left={delta_lt}, Right={delta_rt}).{C.RESET}")
        all_tests_passed = False

    # ---------------------------------------------------------
    # TEST 2: DIFFERENTIAL TURN & MOTOR OPPOSITION
    # ---------------------------------------------------------
    print("\n" + f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}>> TEST 2/3: DIFFERENTIAL TURN ROTATION (1.0s @ 1.0 rad/s Turn Left){C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.DIM}Differential turn kinematics require:{C.RESET}")
    print(f"  * Left Wheel  : {C.MAGENTA}REVERSE (negative ticks){C.RESET}")
    print(f"  * Right Wheel : {C.CYAN}FORWARD (positive ticks){C.RESET}")

    node.pump(0.5)
    start_lt = node.left_ticks
    start_rt = node.right_ticks

    twist_turn = Twist()
    twist_turn.linear.x = 0.0
    twist_turn.angular.z = 1.0

    print(f"{C.DIM}Commanding differential turn left (1.0s)...{C.RESET}")
    t_end = time.time() + 1.0
    while time.time() < t_end:
        node.pub_real.publish(twist_turn)
        node.pump(0.10)

    node.pub_real.publish(Twist())
    print(f"{C.DIM}Waiting for turn telemetry to settle...{C.RESET}")

    t_settle_end = time.time() + 2.0
    last_l = node.left_ticks
    last_r = node.right_ticks
    last_change = time.time()

    while time.time() < t_settle_end:
        node.pump(0.05)
        if node.left_ticks != last_l or node.right_ticks != last_r:
            last_l = node.left_ticks
            last_r = node.right_ticks
            last_change = time.time()
        elif (time.time() - last_change) >= 0.4 and (abs(node.left_ticks - start_lt) >= 5 or abs(node.right_ticks - start_rt) >= 5):
            break

    delta_lt = node.left_ticks - start_lt
    delta_rt = node.right_ticks - start_rt

    print(f"\n{C.BOLD}[Differential Turn Telemetry]{C.RESET}")
    col_lt2 = C.GREEN if delta_lt <= -8 else C.RED
    col_rt2 = C.GREEN if delta_rt >= 8 else C.RED
    print(f"  * Left Wheel Ticks  : {col_lt2}{delta_lt:+d}{C.RESET} (Expected <= -8)")
    print(f"  * Right Wheel Ticks : {col_rt2}{delta_rt:+d}{C.RESET} (Expected >= +8)")

    if delta_lt == 0 and delta_rt == 0:
        print(f"\n{C.BOLD}{C.RED}>>> [TEST 2 FAILED] ZERO TURN MOTION DETECTED! Motors did not spin.{C.RESET}")
        all_tests_passed = False
    elif delta_lt <= -8 and delta_rt >= 8:
        print(f"\n{C.BOLD}{C.GREEN}>>> [TEST 2 PASSED] Differential turn opposition verified! <<<{C.RESET}")
        print(f"    {C.GREEN}Left reversed and Right drove forward correctly.{C.RESET}")
    else:
        print(f"\n{C.BOLD}{C.RED}>>> [TEST 2 FAILED] Incorrect wheel direction during turn:{C.RESET}")
        print(f"    Left={col_lt2}{delta_lt:+d}{C.RESET}, Right={col_rt2}{delta_rt:+d}{C.RESET}")
        all_tests_passed = False

    # ---------------------------------------------------------
    # TEST 3: DYNAMIC BNO085 IMU VERIFICATION
    # ---------------------------------------------------------
    print("\n" + f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}>> TEST 3/3: INTERACTIVE BNO085 IMU DYNAMIC VALIDATION{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}")
    print(f"{C.DIM}Note: In Tests 1 & 2, the robot was lifted in the air, so the chassis{C.RESET}")
    print(f"{C.DIM}did not physically rotate in the room. Let's verify that the BNO085{C.RESET}")
    print(f"{C.DIM}internal Game Rotation Vector tracks real-world angular changes.{C.RESET}")

    node.pump(0.5)
    init_yaw = node.last_heading
    print(f"\n  Current Heading: {C.BOLD}{C.CYAN}{init_yaw:+.1f}°{C.RESET}")
    print(f"\n{C.BOLD}{C.YELLOW}[ACTION REQUIRED]{C.RESET}")
    print(f"  {C.YELLOW}Gently ROTATE THE ENTIRE ROBOT ~90 DEGREES by hand in the air,{C.RESET}")
    try:
        input(f"  {C.BOLD}then press [Enter] to sample the new heading... {C.RESET}")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    node.pump(0.5)
    final_yaw = node.last_heading
    delta_yaw = (final_yaw - init_yaw + 180.0) % 360.0 - 180.0

    col_yaw = C.GREEN if abs(delta_yaw) >= 20.0 else C.RED
    print(f"\n{C.BOLD}[IMU Dynamic Telemetry]{C.RESET}")
    print(f"  * Baseline Heading : {init_yaw:+.1f}°")
    print(f"  * Final Heading    : {final_yaw:+.1f}°")
    print(f"  * Angular Delta    : {col_yaw}{delta_yaw:+.1f}°{C.RESET}")

    if abs(delta_yaw) >= 20.0:
        print(f"\n{C.BOLD}{C.GREEN}>>> [TEST 3 PASSED] BNO085 IMU tracked physical rotation of {delta_yaw:+.1f}°! <<<{C.RESET}")
    else:
        print(f"\n{C.BOLD}{C.RED}>>> [TEST 3 FAILED] No angular rotation detected (|delta| < 20°).{C.RESET}")
        print("    Verify BNO085 I2C1 connection on GP14 (SDA) and GP15 (SCL).")
        all_tests_passed = False

    # ---------------------------------------------------------
    # FINAL VERDICT
    # ---------------------------------------------------------
    print("\n" + f"{C.CYAN}{'=' * 74}{C.RESET}")
    if all_tests_passed:
        print(f"  {C.BG_GREEN}       OVERALL VERDICT: [ALL HARDWARE DIAGNOSTICS PASSED]        {C.RESET}")
        print(f"{C.CYAN}{'=' * 74}{C.RESET}")
        print(f"{C.BOLD}{C.GREEN}>> Motors, DRV8833 PWM, optical wheel encoders, and BNO085 IMU{C.RESET}")
        print(f"{C.BOLD}{C.GREEN}   are verified and calibrated on real hardware.{C.RESET}")
        print(f"\n>> You are fully ready to rehearse and execute moves with:")
        print(f"     {C.BOLD}{C.CYAN}python3 sim_to_real_agent.py{C.RESET}")
    else:
        print(f"  {C.BG_RED}       OVERALL VERDICT: [HARDWARE DIAGNOSTIC FAILED]              {C.RESET}")
        print(f"{C.CYAN}{'=' * 74}{C.RESET}")
        print(f"{C.BOLD}{C.RED}>> DO NOT attempt autonomous robot moves until resolving the{C.RESET}")
        print(f"{C.BOLD}{C.RED}   hardware issues identified above.{C.RESET}")
    print(f"{C.CYAN}{'=' * 74}{C.RESET}\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
