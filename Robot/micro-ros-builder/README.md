# Pico 2 W micro-ROS Static Library Builder

This folder builds the static micro-ROS client archive (`libmicroros.a`) and matching headers for the **Raspberry Pi Pico 2 W** (RP2350 / ARM Cortex-M33).

Unlike the instructor's default setup, **Docker Dev Containers are not used here**. Build tools, ROS 2 packages, and the ARM GNU toolchain are managed natively in WSL via **Pixi** and the Pico SDK in `Robot/pico2w_led/.tools/`.

---

## Why this builder is needed

Micro-ROS for embedded microcontrollers does not use dynamic runtime heap allocation for ROS middleware entities. Static table capacities must be defined at library compile time inside `colcon.meta`:

* `RMW_UXRCE_MAX_NODES=2`
* `RMW_UXRCE_MAX_PUBLISHERS=10`
* `RMW_UXRCE_MAX_SUBSCRIPTIONS=10`
* `RMW_UXRCE_MAX_SERVICES=4`
* `RMW_UXRCE_MAX_CLIENTS=4`
* `RMW_UXRCE_MAX_HISTORY=8`

These generous limits support Module 4 (LED), Module 5 (IMU & Motor Tilt Control), Module 7 (Odometry & PID), and custom multi-topic robot projects without requiring frequent recompilations.

---

## How to Build

### Option 1: Via Pixi Task (Recommended)

From the project root:
```bash
pixi run build-libmicroros
```

### Option 2: Direct Bash Script

Inside the active root Pixi ROS 2 environment:
```bash
bash src/Robot/micro-ros-builder/build-library.sh
```

---

## Output Location

The script automatically publishes its verified artifacts into:

```text
src/Robot/pico2w_led/libmicroros/
├── architecture.txt   # Verified ARMv8-M Mainline / Cortex-M33 object tags
├── include/            # Micro-ROS and RMW C headers
└── libmicroros.a      # Static library linked by CMakeLists.txt
```

Firmware builds in `src/Robot/pico2w_led/` link this archive statically during `pixi run build`.
