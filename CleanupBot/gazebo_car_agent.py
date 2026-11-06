"""ROSA agent that drives the simulated robot car in Gazebo Harmonic, using a local Ollama model.

Run while the simulation is running:
    PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi run python gazebo_car_agent.py
"""

import os
import sys

# Ensure local directory is on python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from langchain_ollama import ChatOllama
from rosa import ROSA, RobotSystemPrompts

import gazebo_car_tools as car

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = f"http://{OLLAMA_URL}"


def build_prompts() -> RobotSystemPrompts:
    return RobotSystemPrompts(
        embodiment_and_persona=(
            "You are the driver of a small two-wheel differential-drive robot car in a "
            "Gazebo simulation. You move it only with the robot tools provided."
        ),
        about_your_environment=(
            "The robot starts at (0, 0) facing +x in a walled arena with box obstacles "
            "and a green goal pad. The world map is:\n" + car.describe_world()
        ),
        about_your_capabilities=(
            "get_robot_status reports position, heading, IMU, and camera status. "
            "go_to(x, y) drives to a point and go_to_place(place) drives to a named place "
            "such as 'goal' or 'home'; both plan a route around obstacles automatically. "
            "drive_forward(meters) drives forward in a straight line. "
            "turn_left(degrees) and turn_right(degrees) turn in place. "
            "save_camera_image saves a camera picture and reports the colors in view. "
            "stop_robot stops the robot."
        ),
        critical_instructions=(
            "For each request, choose the single robot tool that matches it and call it "
            "once. The tool result is shown to the human directly. "
            "Never claim the robot moved unless a tool reported it."
        ),
        constraints_and_guardrails=(
            "Use only coordinates the human gives you or places from the world map. "
            "If the human asks for several moves at once, do only the first move and "
            "tell them to send the next move after it finishes."
        ),
    )


class GazeboCarAgent(ROSA):
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
                car.get_robot_status,
                car.go_to,
                car.go_to_place,
                car.drive_forward,
                car.turn_left,
                car.turn_right,
                car.stop_robot,
                car.save_camera_image,
            ],
            prompts=build_prompts(),
            verbose=True,
            streaming=False,
            accumulate_chat_history=False,
            max_iterations=12,
        )


def main() -> None:
    car.get_bridge()
    agent = GazeboCarAgent()
    print(f"Gazebo car agent ready (model: {OLLAMA_MODEL}). Type 'quit' to exit.")
    try:
        while True:
            try:
                request = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if request.lower() in ("quit", "exit"):
                break
            if not request:
                continue
            try:
                print(f"ROSA> {agent.invoke(request)}")
            except KeyboardInterrupt:
                car.get_bridge().stop()
                print("\nInterrupted. The robot was stopped.")
    finally:
        car.close_bridge()


if __name__ == "__main__":
    main()
