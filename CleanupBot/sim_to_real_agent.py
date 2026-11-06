"""Sim-to-Real ROSA Agent for CleanupBot in ROS 2 Jazzy & Gazebo Harmonic.

Runs with local Ollama:
    PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi run python sim_to_real_agent.py
"""

from __future__ import annotations

import os
import sys

# Ensure local directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from langchain_ollama import ChatOllama
from rosa import ROSA, RobotSystemPrompts

import sim_to_real_tools as s2r


OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:cloud")  # "llama3.2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = f"http://{OLLAMA_URL}"


def build_system_prompts() -> RobotSystemPrompts:
    return RobotSystemPrompts(
        embodiment_and_persona=(
            "You are the embodied AI driver for CleanupBot. You control both a simulated digital "
            "twin in Gazebo Harmonic and a physical differential-drive robot car equipped with an "
            "OWI-535 gripper and eye-in-hand BNO085 IMU. You operate under strict sim-to-real "
            "safety guidelines."
        ),
        about_your_environment=(
            "In simulation, the robot operates in a custom apartment digital twin starting at (0, 0) facing +X. "
            "A small toy wooden box is located directly ahead at X=0.50m (Y=0.0m), and a fidget spinner is located "
            "at X=0.48m, Y=-0.15m (15cm to the right). A red box / larger obstacles are also present. In reality, "
            "the robot operates on an indoor floor where linear distance is measured via tape measure and turns are "
            "measured via the real BNO085 IMU heading."
        ),
        about_your_capabilities=(
            "1. rehearse_move(action, magnitude, speed): Rehearses any plain-English move (e.g. forward, "
            "turn_left, turn_right) in Gazebo Harmonic first. If the simulation detects an obstacle "
            "collision, it automatically rejects the move and refuses to offer it on the real robot. "
            "If the rehearsal passes cleanly, it requests operator approval before driving the physical robot.\n"
            "2. emergency_stop(): Immediately stops all motors on both simulation and physical robot.\n"
            "3. get_comparison_table(): Displays the Markdown sim-to-real measurement table."
        ),
        critical_instructions=(
            "For every user request involving motion, call rehearse_move with the appropriate action and "
            "magnitude. NEVER attempt to bypass the simulation rehearsal. Always explain the simulated "
            "result to the user. When asked to show the comparison table or wrap up, call get_comparison_table."
        ),
        constraints_and_guardrails=(
            "Maximum linear speed is 0.30 m/s; maximum angular speed is 1.50 rad/s; maximum single move duration "
            "is 10 seconds. Never execute on physical hardware without successful rehearsal and human approval."
        ),
    )


class SimToRealAgent(ROSA):
    """ROSA agent coordinating Gazebo rehearsal and hardware execution."""

    def __init__(self) -> None:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            num_ctx=16384,
            base_url=OLLAMA_URL,
        )
        super().__init__(
            ros_version=2,
            llm=llm,
            tools=[
                s2r.rehearse_move,
                s2r.emergency_stop,
                s2r.get_comparison_table,
            ],
            prompts=build_system_prompts(),
            verbose=True,
            streaming=False,
            accumulate_chat_history=False,
            max_iterations=8,
        )


def main() -> None:
    print("=" * 70)
    print("  CleanupBot Sim-to-Real Rehearsal Agent")
    print(f"  Model: {OLLAMA_MODEL} via {OLLAMA_URL}")
    print("=" * 70)
    print("Commands you can test:")
    print("  - 'Drive forward 40 centimeters'  (Clean rehearsal -> asks approval -> runs real)")
    print("  - 'Turn left 90 degrees'          (Clean turn -> asks approval -> runs real)")
    print("  - 'Drive forward 1.5 meters'      (Hits red box in Gazebo -> Auto-blocked!)")
    print("  - 'Show comparison table'         (Prints sim-to-real comparison records)")
    print("Type 'quit' or 'exit' to terminate.\n")

    agent = SimToRealAgent()

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        try:
            response = agent.invoke(user_input)
            print(f"\nROSA> {response}")
        except Exception as exc:
            print(f"\nROSA [Error]> {exc}")


if __name__ == "__main__":
    main()
