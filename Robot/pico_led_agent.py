"""Standalone ROSA agent targeting the Pico 2 W onboard LED."""

import sys
from langchain_ollama import ChatOllama
from rosa import ROSA, RobotSystemPrompts
from pico_led_tool import set_pico_led


def build_agent() -> ROSA:
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434",
    )

    prompts = RobotSystemPrompts(
        embodiment_and_persona=(
            "You control the onboard LED of a Raspberry Pi Pico 2 W microcontroller via ROS 2. "
            "Always use the set_pico_led tool when asked to turn the LED on or off. "
            "Call set_pico_led with a flat boolean argument: set_pico_led(state=True) to turn on, set_pico_led(state=False) to turn off. "
            "`wait` commands must use integer seconds, e.g., wait(2) to wait for 2 seconds. "
            "Report success only when the tool returns an explicit hardware acknowledgment. "
            "If an error or timeout occurs, report the error verbatim and never claim success."
        )
    )

    return ROSA(
        ros_version=2,
        llm=llm,
        tools=[set_pico_led],
        prompts=prompts,
        max_iterations=5,
        verbose=True,
    )


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = "Turn on the Pico LED for 2 seconds as a hello-world test. Then turn it off for 2 seconds."

    agent = build_agent()
    response = agent.invoke(prompt)
    print("\n--- Agent Response ---")
    print(response)


if __name__ == "__main__":
    main()