"""ROSA tools for Sim-to-Real rehearsal, safety gating, and comparison.

Provides:
  - rehearse_move: Rehearses plain-English moves in Gazebo Harmonic first.
    If the simulated robot is blocked / collides with an obstacle, it automatically
    REJECTS the move and refuses to offer physical execution.
    If safe, it prompts the operator for approval before driving the physical CleanupBot.
  - emergency_stop: Immediately cuts power and halts both robots.
  - get_comparison_table: Formats the sim-to-real measurement table with encoder and tape measure data.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from robot_motion import RobotMotion, MotionResult

SCRIPT_DIR = Path(__file__).resolve().parent
MD_REPORT_PATH = SCRIPT_DIR / "sim_to_real_comparison.md"
HTML_REPORT_PATH = SCRIPT_DIR / "sim_to_real_comparison.html"


# Singleton motion runner
_motion_runner = RobotMotion()

@dataclass
class LoggedMove:
    command_desc: str
    cmd_linear: float
    cmd_angular: float
    duration_s: float
    sim_dist_m: float
    sim_turn_deg: float
    sim_blocked: bool
    approved_by_human: bool
    real_encoder_m: Optional[float]
    real_tape_m: Optional[float]
    real_turn_deg: Optional[float]


_move_history: List[LoggedMove] = []


class RehearseMoveArgs(BaseModel):
    action: str = Field(
        description="Type of move: 'forward', 'backward', 'turn_left', 'turn_right', or 'custom'.",
    )
    magnitude: float = Field(
        description="Amount to move: distance in meters for forward/backward, or angle in degrees for turns.",
    )
    speed: float = Field(
        default=0.15,
        description="Speed: linear speed in m/s (max 0.30) or angular speed in rad/s (max 1.20).",
    )


def _compute_motion_params(action: str, magnitude: float, speed: float) -> tuple[float, float, float, float, float]:
    action = action.lower().strip()
    mag = abs(float(magnitude))

    # If speed is <= 0.01 (e.g. 0 was passed or defaulted), use calibrated nominal speed (0.15 m/s)
    if speed <= 0.01:
        speed = 0.15
    else:
        speed = max(0.08, min(0.25, abs(float(speed))))

    target_dist = 0.0
    target_angle = 0.0

    if action in ("forward", "straight", "drive_forward"):
        linear = speed
        angular = 0.0
        target_dist = mag
        duration = max(0.5, min(10.0, target_dist / linear))
    elif action in ("backward", "reverse"):
        linear = -speed
        angular = 0.0
        target_dist = mag
        duration = max(0.5, min(10.0, target_dist / speed))
    elif action in ("turn_left", "left"):
        linear = 0.0
        # Optimal 0.80 rad/s (~46 deg/s) for high torque and minimal wheel slip
        angular = 0.80
        target_angle = mag
        turn_rad = math.radians(target_angle)
        duration = max(0.4, min(10.0, turn_rad / angular))
    elif action in ("turn_right", "right"):
        linear = 0.0
        angular = -0.80
        target_angle = mag
        turn_rad = math.radians(target_angle)
        duration = max(0.4, min(10.0, turn_rad / abs(angular)))
    else:
        linear = speed
        angular = 0.0
        target_dist = mag
        duration = max(0.5, min(10.0, target_dist / speed if speed > 0 else 1.0))

    return linear, angular, duration, target_dist, target_angle


@tool(args_schema=RehearseMoveArgs)
def rehearse_move(action: str, magnitude: float, speed: float = 0.15) -> str:
    """
    Rehearse a move in Gazebo simulation first.
    If the simulated robot hits an obstacle, this tool BLOCKS the move and prevents
    it from being offered or executed on the real robot.
    If safe, it prompts the operator in console for [yes/no] approval before driving
    the physical CleanupBot.
    """
    linear, angular, duration, target_dist, target_angle = _compute_motion_params(action, magnitude, speed)
    desc = f"{action} ({magnitude:.2f} {'m' if 'turn' not in action else '°'})"

    print("\n" + "=" * 65)
    print(f"  [SIMULATION REHEARSAL] Rehearsing {desc} in Gazebo Harmonic...")
    print(f"  Parameters: Linear={linear:.2f} m/s, Angular={angular:.2f} rad/s, Duration={duration:.2f} s")
    print("=" * 65)

    # 1. Rehearse in Gazebo Harmonic
    sim_result: MotionResult = _motion_runner.move(
        target="sim",
        linear=linear,
        angular=angular,
        seconds=duration,
        target_distance=target_dist,
        target_angle_deg=target_angle,
    )
    print(sim_result.summary())

    # 2. Obstacle Collision / Block Check
    if sim_result.possibly_blocked:
        print("\n" + "!" * 65)
        print("  [CRITICAL SAFETY BLOCK] REHEARSAL FAILED!")
        print(f"  Gazebo CleanupBot hit an obstacle (traveled only {sim_result.distance_m:.2f}m).")
        print("  Movement rejected for safety — will NOT be offered or run on real robot.")
        print("!" * 65 + "\n")

        _move_history.append(
            LoggedMove(
                command_desc=desc,
                cmd_linear=linear,
                cmd_angular=angular,
                duration_s=duration,
                sim_dist_m=sim_result.distance_m,
                sim_turn_deg=sim_result.turn_deg,
                sim_blocked=True,
                approved_by_human=False,
                real_encoder_m=None,
                real_tape_m=None,
                real_turn_deg=None,
            )
        )
        _export_comparison_files()
        return (
            f"[SAFETY REJECTION] Gazebo rehearsal detected an obstacle collision at {sim_result.distance_m:.2f}m. "
            "Per safety protocols, this move is completely aborted and cannot be run on the real robot."
        )

    # 3. Rehearsal Safe: Human Approval
    print("\n" + "*" * 65)
    print(f"  [REHEARSAL PASSED] Move completed cleanly in simulation:")
    print(f"    - Simulated Distance : {sim_result.distance_m:.3f} m")
    print(f"    - Simulated Turn     : {sim_result.turn_deg:.1f}°")
    print("*" * 65)

    approval_input = ""
    try:
        approval_input = input(
            "\n>> Do you approve running this exact move on your real CleanupBot? (yes/no): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        approval_input = "no"

    if approval_input not in ("yes", "y"):
        print("\n[OPERATOR REJECTED] Move rejected by human operator. Real robot stayed stationary.\n")
        _move_history.append(
            LoggedMove(
                command_desc=desc,
                cmd_linear=linear,
                cmd_angular=angular,
                duration_s=duration,
                sim_dist_m=sim_result.distance_m,
                sim_turn_deg=sim_result.turn_deg,
                sim_blocked=False,
                approved_by_human=False,
                real_encoder_m=None,
                real_tape_m=None,
                real_turn_deg=None,
            )
        )
        _export_comparison_files()
        return (
            f"[OPERATOR REJECTED] Rehearsal in Gazebo succeeded ({sim_result.distance_m:.2f}m, {sim_result.turn_deg:.1f}°), "
            "but human operator rejected physical execution. Real robot was not moved."
        )

    # 4. Hardware Execution
    print("\n" + "=" * 65)
    print("  [HARDWARE EXECUTION] Executing approved move on real CleanupBot...")
    print("  Watch the physical robot; keep hands ready for Ctrl+C emergency stop.")
    print("=" * 65)

    real_result: MotionResult = _motion_runner.move(
        target="real",
        linear=linear,
        angular=angular,
        seconds=duration,
        target_distance=target_dist,
        target_angle_deg=target_angle,
    )
    print(real_result.summary())

    # 5. Measure Tape Distance & Compare with Encoders
    measured_tape = None
    if abs(linear) > 0.01:
        default_val = real_result.encoder_distance_m if real_result.encoder_distance_m > 0.001 else real_result.distance_m
        try:
            val_str = input(
                f"\n>> [PHYSICAL MEASUREMENT] Enter actual distance measured with tape measure (in meters, default {default_val:.2f}): "
            ).strip()
            measured_tape = float(val_str) if val_str else default_val
        except (ValueError, EOFError, KeyboardInterrupt):
            measured_tape = default_val
    else:
        measured_tape = 0.0

    _move_history.append(
        LoggedMove(
            command_desc=desc,
            cmd_linear=linear,
            cmd_angular=angular,
            duration_s=duration,
            sim_dist_m=sim_result.distance_m,
            sim_turn_deg=sim_result.turn_deg,
            sim_blocked=False,
            approved_by_human=True,
            real_encoder_m=real_result.encoder_distance_m,
            real_tape_m=measured_tape,
            real_turn_deg=real_result.turn_deg,
        )
    )

    _export_comparison_files()
    print(f"\n✓ Physical move completed and logged to: {MD_REPORT_PATH.name} and {HTML_REPORT_PATH.name}")
    return (
        f"[SIM-TO-REAL EXECUTED] Approved move '{desc}' executed on both robots.\n"
        f"  - Commanded: v={linear:.2f}m/s, w={angular:.2f}rad/s, t={duration:.2f}s\n"
        f"  - Gazebo Sim: Distance={sim_result.distance_m:.3f}m, Turn={sim_result.turn_deg:.1f}°\n"
        f"  - Real Encoders: Distance={real_result.encoder_distance_m:.3f}m, Turn={real_result.turn_deg:.1f}° (BNO085 IMU)\n"
        f"  - Tape Measure: Distance={measured_tape:.3f}m"
    )


class EmptyArgs(BaseModel):
    pass


@tool(args_schema=EmptyArgs)
def emergency_stop() -> str:
    """Immediately stop both the Gazebo simulated robot and the real robot."""
    _motion_runner.stop("both")
    return "[EMERGENCY STOP] All motors stopped immediately on simulation and hardware."


def _export_comparison_files() -> tuple[Path, Path]:
    """Generates clean Markdown report and interactive HTML dashboard."""
    if not _move_history:
        return MD_REPORT_PATH, HTML_REPORT_PATH

    total = len(_move_history)
    blocked = sum(1 for m in _move_history if m.sim_blocked)
    approved = sum(1 for m in _move_history if m.approved_by_human)
    human_rej = sum(1 for m in _move_history if not m.sim_blocked and not m.approved_by_human)

    # -------------------------------------------------------------
    # 1. MARKDOWN COMPARISON REPORT (.md)
    # -------------------------------------------------------------
    md_lines = [
        "# CleanupBot: Sim-to-Real Telemetry Comparison Report",
        f"**Module 6 Demonstration Record** | *Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary Metrics",
        f"- **Total Rehearsals**: {total}",
        f"- **Safety Auto-Rejections (Collision Gating)**: {blocked}",
        f"- **Operator Approved & Executed**: {approved}",
        f"- **Operator Manual Rejections**: {human_rej}",
        "",
        "## Sim-to-Real Telemetry Comparison Table",
        "",
        "| Move / Command | Commanded $(v, \\omega, t)$ | Gazebo Sim | Real Encoders | Real Tape & IMU | Status | Sim-to-Real Gap ($\\Delta$) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m in _move_history:
        cmd_str = f"v={m.cmd_linear:.2f}m/s, w={m.cmd_angular:.2f}r/s, t={m.duration_s:.1f}s"
        sim_str = f"{m.sim_dist_m:.3f}m, {m.sim_turn_deg:.1f}°"
        if m.sim_blocked:
            enc_str = "—"
            real_str = "—"
            status_badge = "**[BLOCKED] Auto-Rejected**"
            gap_str = "Obstacle detected in sim; physical move inhibited."
        elif not m.approved_by_human:
            enc_str = "—"
            real_str = "—"
            status_badge = "**[REJECTED] Operator Manual**"
            gap_str = "Human operator declined execution."
        else:
            enc_str = f"{m.real_encoder_m:.3f}m" if m.real_encoder_m is not None else "—"
            real_str = f"{m.real_tape_m:.3f}m, {m.real_turn_deg:.1f}°"
            status_badge = "**[APPROVED] Executed**"
            d_dist = abs(m.sim_dist_m - m.real_tape_m) if m.real_tape_m is not None else 0.0
            d_turn = abs(m.sim_turn_deg - m.real_turn_deg) if m.real_turn_deg is not None else 0.0
            gap_str = f"$\\Delta$Dist={d_dist:.3f}m, $\\Delta$Turn={d_turn:.1f}°"

        md_lines.append(
            f"| **{m.command_desc}** | {cmd_str} | {sim_str} | {enc_str} | {real_str} | {status_badge} | {gap_str} |"
        )

    md_lines.extend([
        "",
        "## Visual Accuracy Breakdown",
        "",
    ])

    for i, m in enumerate(_move_history, 1):
        md_lines.append(f"### Move {i}: {m.command_desc}")
        if m.sim_blocked:
            md_lines.extend([
                f"- **Result**: 🛑 **Blocked by Collision in Gazebo** (Traveled only {m.sim_dist_m:.2f}m before obstacle contact).",
                "- **Safety Action**: Real robot execution automatically prohibited.",
            ])
        elif not m.approved_by_human:
            md_lines.append("- **Result**: ⏸️ Rehearsal succeeded; execution declined by operator.")
        else:
            if "turn" in m.command_desc.lower():
                max_ang = max(1.0, abs(m.sim_turn_deg), abs(m.real_turn_deg or 0.0))
                s_bar = "█" * int(24 * (abs(m.sim_turn_deg) / max_ang))
                r_bar = "█" * int(24 * (abs(m.real_turn_deg or 0.0) / max_ang))
                md_lines.extend([
                    "```text",
                    f"  Gazebo Turn:    [{s_bar:<24}] {m.sim_turn_deg:+.1f}°",
                    f"  Real BNO085:    [{r_bar:<24}] {m.real_turn_deg:+.1f}° (IMU Planar Yaw)",
                    f"  Angular Delta:  {abs(m.sim_turn_deg - (m.real_turn_deg or 0.0)):.1f}°",
                    "```",
                ])
            else:
                max_d = max(0.01, m.sim_dist_m, m.real_tape_m or 0.0, m.real_encoder_m or 0.0)
                s_bar = "█" * int(24 * (m.sim_dist_m / max_d))
                e_bar = "█" * int(24 * ((m.real_encoder_m or 0.0) / max_d))
                t_bar = "█" * int(24 * ((m.real_tape_m or 0.0) / max_d))
                md_lines.extend([
                    "```text",
                    f"  Gazebo Sim:     [{s_bar:<24}] {m.sim_dist_m:.3f} m",
                    f"  Real Encoders:  [{e_bar:<24}] {m.real_encoder_m or 0.0:.3f} m",
                    f"  Real Tape Meas: [{t_bar:<24}] {m.real_tape_m or 0.0:.3f} m",
                    f"  Distance Delta: {abs(m.sim_dist_m - (m.real_tape_m or 0.0)):.3f} m",
                    f"  Heading Drift:  {m.real_turn_deg or 0.0:+.1f}° (BNO085 Gyro Lock)",
                    "```",
                ])
        md_lines.append("")

    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    # -------------------------------------------------------------
    # 2. INTERACTIVE HTML DASHBOARD (.html)
    # -------------------------------------------------------------
    html_cards = []
    html_rows = []

    for i, m in enumerate(_move_history, 1):
        cmd_str = f"v={m.cmd_linear:.2f} m/s, w={m.cmd_angular:.2f} rad/s, t={m.duration_s:.1f} s"
        sim_str = f"{m.sim_dist_m:.3f} m, {m.sim_turn_deg:.1f}°"
        if m.sim_blocked:
            badge = '<span class="badge badge-blocked">BLOCKED (Auto-Rejected)</span>'
            enc_str = "—"
            real_str = "—"
            analysis = "Collision detected in simulation; physical move inhibited."
            card_content = f"""
                <div class="result-alert alert-blocked">
                    <strong>🛑 Collision Safety Gate Triggered:</strong> Robot contacted obstacle at {m.sim_dist_m:.2f} m. Hardware execution was safely refused.
                </div>
            """
        elif not m.approved_by_human:
            badge = '<span class="badge badge-rejected">MANUAL REJECT</span>'
            enc_str = "—"
            real_str = "—"
            analysis = "Rehearsal passed cleanly; human operator declined physical run."
            card_content = """
                <div class="result-alert alert-rejected">
                    <strong>⏸️ Operator Manual Reject:</strong> Simulation passed, but physical execution was declined.
                </div>
            """
        else:
            badge = '<span class="badge badge-approved">APPROVED (Executed)</span>'
            enc_str = f"{m.real_encoder_m:.3f} m" if m.real_encoder_m is not None else "—"
            real_str = f"{m.real_tape_m:.3f} m, {m.real_turn_deg:.1f}°"
            d_dist = abs(m.sim_dist_m - m.real_tape_m) if m.real_tape_m is not None else 0.0
            d_turn = abs(m.sim_turn_deg - m.real_turn_deg) if m.real_turn_deg is not None else 0.0
            analysis = f"ΔDist = {d_dist*1000:.0f} mm, ΔTurn = {d_turn:.1f}°"

            if "turn" in m.command_desc.lower():
                max_ang = max(1.0, abs(m.sim_turn_deg), abs(m.real_turn_deg or 0.0))
                s_pct = min(100, int(100 * abs(m.sim_turn_deg) / max_ang))
                r_pct = min(100, int(100 * abs(m.real_turn_deg or 0.0) / max_ang))
                card_content = f"""
                    <div class="bar-group">
                        <div class="bar-label"><span>Gazebo Turn</span><span>{m.sim_turn_deg:+.1f}°</span></div>
                        <div class="bar-track"><div class="bar-fill bar-sim" style="width: {s_pct}%;"></div></div>
                    </div>
                    <div class="bar-group">
                        <div class="bar-label"><span>Real BNO085 IMU</span><span>{m.real_turn_deg:+.1f}°</span></div>
                        <div class="bar-track"><div class="bar-fill bar-real" style="width: {r_pct}%;"></div></div>
                    </div>
                    <div class="delta-tag">Angular Discrepancy: {d_turn:.1f}°</div>
                """
            else:
                max_d = max(0.01, m.sim_dist_m, m.real_tape_m or 0.0, m.real_encoder_m or 0.0)
                s_pct = min(100, int(100 * m.sim_dist_m / max_d))
                e_pct = min(100, int(100 * (m.real_encoder_m or 0.0) / max_d))
                t_pct = min(100, int(100 * (m.real_tape_m or 0.0) / max_d))
                card_content = f"""
                    <div class="bar-group">
                        <div class="bar-label"><span>Gazebo Sim</span><span>{m.sim_dist_m:.3f} m</span></div>
                        <div class="bar-track"><div class="bar-fill bar-sim" style="width: {s_pct}%;"></div></div>
                    </div>
                    <div class="bar-group">
                        <div class="bar-label"><span>Real Encoders</span><span>{m.real_encoder_m or 0.0:.3f} m</span></div>
                        <div class="bar-track"><div class="bar-fill bar-encoder" style="width: {e_pct}%;"></div></div>
                    </div>
                    <div class="bar-group">
                        <div class="bar-label"><span>Real Ground-Truth Tape</span><span>{m.real_tape_m or 0.0:.3f} m</span></div>
                        <div class="bar-track"><div class="bar-fill bar-real" style="width: {t_pct}%;"></div></div>
                    </div>
                    <div class="delta-tag">Distance Delta: {d_dist*1000:.0f} mm | Gyro Heading Drift: {m.real_turn_deg or 0.0:+.1f}°</div>
                """

        html_rows.append(f"""
            <tr>
                <td><strong>{m.command_desc}</strong></td>
                <td><code>{cmd_str}</code></td>
                <td>{sim_str}</td>
                <td>{enc_str}</td>
                <td>{real_str}</td>
                <td>{badge}</td>
                <td>{analysis}</td>
            </tr>
        """)

        html_cards.append(f"""
            <div class="move-card">
                <div class="move-card-header">
                    <h4>Move {i}: {m.command_desc}</h4>
                    {badge}
                </div>
                <div class="move-card-body">
                    {card_content}
                </div>
            </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CleanupBot: Sim-to-Real Telemetry Dashboard</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #151e32;
            --surface-hover: #1c2842;
            --border: #23304c;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #10b981;
            --accent-purple: #a855f7;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 2.5rem;
            line-height: 1.5;
        }}
        .header {{ margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }}
        .header h1 {{ font-size: 2rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.75rem; }}
        .header p {{ color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.25rem; }}
        .stat-card .title {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }}
        .stat-card .val {{ font-size: 1.85rem; font-weight: 700; margin-top: 0.25rem; }}
        .val-blue {{ color: var(--accent-blue); }}
        .val-green {{ color: var(--accent-green); }}
        .val-red {{ color: var(--accent-red); }}
        .section-title {{ font-size: 1.25rem; font-weight: 600; margin: 2rem 0 1rem; color: #fff; display: flex; align-items: center; gap: 0.5rem; }}
        .table-wrap {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.75rem; overflow-x: auto; margin-bottom: 2.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        th {{ background: #1a253e; color: var(--text-muted); font-weight: 600; padding: 0.85rem 1rem; border-bottom: 1px solid var(--border); text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
        td {{ padding: 1rem; border-bottom: 1px solid var(--border); }}
        tr:hover td {{ background: var(--surface-hover); }}
        code {{ background: #0b0f19; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.8rem; color: var(--accent-blue); border: 1px solid var(--border); }}
        .badge {{ display: inline-flex; align-items: center; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }}
        .badge-approved {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-blocked {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-rejected {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1.25rem; }}
        .move-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.25rem; }}
        .move-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.75rem; }}
        .move-card-header h4 {{ font-size: 1rem; color: #fff; }}
        .bar-group {{ margin-bottom: 0.75rem; }}
        .bar-label {{ display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.25rem; }}
        .bar-track {{ background: #0b0f19; height: 10px; border-radius: 5px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }}
        .bar-fill {{ height: 100%; border-radius: 5px; transition: width 0.6s ease; }}
        .bar-sim {{ background: linear-gradient(90deg, #0284c7, #38bdf8); }}
        .bar-encoder {{ background: linear-gradient(90deg, #7c3aed, #a855f7); }}
        .bar-real {{ background: linear-gradient(90deg, #059669, #10b981); }}
        .delta-tag {{ font-size: 0.75rem; color: var(--text-muted); background: #0b0f19; padding: 0.35rem 0.65rem; border-radius: 0.4rem; display: inline-block; margin-top: 0.5rem; border: 1px solid var(--border); }}
        .result-alert {{ padding: 0.75rem 1rem; border-radius: 0.5rem; font-size: 0.85rem; }}
        .alert-blocked {{ background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #fca5a5; }}
        .alert-rejected {{ background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); color: #fde68a; }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 0.8rem; margin-top: 3rem; border-top: 1px solid var(--border); padding-top: 1.5rem; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 CleanupBot: Sim-to-Real Telemetry Verification</h1>
        <p>Module 6 Lab Record • Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Total Rehearsals</div>
            <div class="val val-blue">{total}</div>
        </div>
        <div class="stat-card">
            <div class="title">Safety Auto-Rejections</div>
            <div class="val val-red">{blocked}</div>
        </div>
        <div class="stat-card">
            <div class="title">Approved Hardware Runs</div>
            <div class="val val-green">{approved}</div>
        </div>
    </div>

    <div class="section-title">📊 Complete Rehearsal & Telemetry Comparison Table</div>
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Move / Action</th>
                    <th>Commanded</th>
                    <th>Gazebo Twin</th>
                    <th>Real Encoders</th>
                    <th>Real Tape & IMU</th>
                    <th>Safety Gate</th>
                    <th>Sim-to-Real Analysis</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>

    <div class="section-title">📈 Visual Accuracy & Delta Breakdown</div>
    <div class="cards-grid">
        {''.join(html_cards)}
    </div>

    <div class="footer">
        CleanupBot Sim-to-Real Verification Suite • ROS 2 Jazzy & Gazebo Harmonic • FastMCP & ROSA Agent
    </div>
</body>
</html>"""

    HTML_REPORT_PATH.write_text(html_content, encoding="utf-8")
    return MD_REPORT_PATH, HTML_REPORT_PATH


@tool(args_schema=EmptyArgs)
def get_comparison_table() -> str:
    """Generate and return the Markdown sim-to-real comparison table for all executed moves."""
    if not _move_history:
        return "No moves have been executed yet."

    md_path, html_path = _export_comparison_files()

    headers = [
        "| Move / Command | Commanded | Gazebo Sim | Real Encoders | Real Tape & IMU | Status | Sim-to-Real Analysis |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    rows = []
    for m in _move_history:
        cmd_str = f"{m.cmd_linear:.2f}m/s, {m.cmd_angular:.2f}r/s, {m.duration_s:.1f}s"
        sim_str = f"{m.sim_dist_m:.2f}m, {m.sim_turn_deg:.1f}°"
        if m.sim_blocked:
            enc_str = "N/A"
            real_str = "N/A (Blocked)"
            appr_str = "Rejected (Auto)"
            gap_str = "Hit obstacle in sim; physical move blocked."
        elif not m.approved_by_human:
            enc_str = "N/A"
            real_str = "N/A (Rejected)"
            appr_str = "Rejected (Human)"
            gap_str = "Human rejected hardware execution."
        else:
            enc_str = f"{m.real_encoder_m:.2f}m" if m.real_encoder_m else "N/A"
            real_str = f"{m.real_tape_m:.2f}m, {m.real_turn_deg:.1f}°"
            appr_str = "Approved"
            dist_diff = abs(m.sim_dist_m - m.real_tape_m)
            turn_diff = abs(m.sim_turn_deg - m.real_turn_deg)
            gap_str = f"ΔDist={dist_diff:.2f}m, ΔTurn={turn_diff:.1f}°"

        rows.append(
            f"| {m.command_desc} | {cmd_str} | {sim_str} | {enc_str} | {real_str} | {appr_str} | {gap_str} |"
        )

    table_md = "\n".join(headers + rows)
    return (
        f"{table_md}\n\n"
        f"✓ Full report saved to: `{md_path.name}`\n"
        f"✓ Visual dashboard saved to: `{html_path.name}` (open in browser for presentation)"
    )
