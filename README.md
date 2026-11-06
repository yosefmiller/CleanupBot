# 2026 Embodied Agentic AI - Workspace Repository

This repository contains the software, firmware, simulations, and hardware schematics for the **Embodied Agentic AI** course, developed and maintained by **Yosef Miller**.

---

## Workspace Structure

```graphql
src/
├── pixi.toml / pixi.lock           # Root course environment (ROS 2 Jazzy, Gazebo Harmonic, ROSA)
├── relay.py                        # High-performance micro-ROS UDP relay (Windows <-> WSL2)
│
├── CleanupBot/                     # Primary Course Semester Project
│   ├── cleanupbot_agent.py         # Autonomous ROSA agent (LangChain + Ollama + MCP)
│   ├── cleanupbot_tools.py         # Robot manipulation & navigation tool primitives
│   ├── cleanupbot_gui.py           # Real-time OpenCV HUD visual interface
│   ├── config.py                   # Central source of truth for kinematics & topics
│   ├── robot_motion.py             # Reusable differential drive safety motion controller
│   ├── ros2_bridge/                # ROS 2 Python client abstraction layer (pub/sub nodes)
│   ├── esp32s3_cam/                # Gripper-mounted ESP32-S3 camera streamer & ROS 2 node
│   ├── VisionMCP/                  # FastMCP vision perception server (YOLO-World + CUDA)
│   ├── MemoryMCP/                  # FastMCP episodic memory server (ChromaDB adaptation)
│   ├── ros2_ws/                    # Gazebo Harmonic apartment simulation workspace
│   ├── gazebo_car_agent.py & tools # Gazebo simulation exploration subsystem
│   ├── sim_to_real_agent.py & tools# Sim-to-real evaluation and validation subsystem
│   ├── tests/                      # Assertive hardware diagnostic & calibration suite
│   └── README.md                   # Comprehensive CleanupBot technical documentation
│
├── Robot/                          # Coursework Module Exercises Sandbox
│   ├── MCP/                        # Coursework MCP servers (datasheet, memory, weather)
│   ├── micro-ros-builder/          # Static libmicroros builder for RP2350
│   ├── pico2w_led/                 # Pico 2 W C firmware (DRV8833, BNO085 IMU, TT Encoders)
│   └── rosa_*.py / pico_*.py       # Modules 2, 4, and 5 lab scripts & visualizers
│
└── Fritzing/                       # Electrical schematics, breadboard layouts, & custom parts
    ├── cleanup_bot.fzz             # Complete robot wiring schematic
    └── cleanup_bot_bb.png          # Visual breadboard hookup diagram
```

---

## Architecture: Coursework vs. Project

* **`src/Robot/` (Coursework Sandbox)**:
  Mirrors the upstream course assignments from the professor's modules (`en665715-embodied-ai`). Use this directory when running homework exercises (e.g. basic ROSA prompt testing, MCP server integration, Pico tilt control experiments).
* **`src/CleanupBot/` (Autonomous Robot Project)**:
  The end-to-end semester project robot. Contains the complete integrated stack: vision perception, episodic memory, closed-loop manipulation, GUI HUD, Gazebo simulation, and assertive diagnostics. See [`CleanupBot/README.md`](CleanupBot/README.md) for full project details.

---

## Getting Started

### 1. Unified Environment
Install all dependencies using Pixi from the `src/` directory:
```bash
cd src
pixi install
```

### 2. Physical Telemetry Relay (Windows Host)
When communicating with the physical robot over the Windows 2.4 GHz Mobile Hotspot:
```bash
# Run in Windows command prompt or PowerShell:
python relay.py
```
This multiplexes UDP packets between the Pico 2 W (`192.168.137.1:8888`) and the WSL2 micro-ROS agent (`127.0.0.1:8889`) with $< 0.1\text{ ms}$ latency.

### 3. micro-ROS Agent (WSL2)
```bash
pixi run micro-ros-agent
```

### 4. Running the CleanupBot Project
Refer to [`CleanupBot/README.md`](CleanupBot/README.md) for detailed launch sequences, simulation workflows, and hardware calibration tests.
