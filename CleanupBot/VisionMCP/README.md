# VisionMCP: FastMCP Vision & Spatial Perception Server

A high-performance FastMCP vision server designed for mobile ground robots in the Embodied AI course. Couples **open-vocabulary YOLO-World** object detection and **CIELAB color identification** with **inverse perspective ground projection** (pixel rays $\to$ metric $(X, Y)$ ground coordinates in meters) and **TypeSafe Jev System One VLM** visual reasoning.

VisionMCP communicates via standard stdio JSON-RPC, making it plug-and-play with any local or remote LLM/VLM agent (e.g., Qwen2.5, Claude, GPT-4, Llama).

---

## Hardware Architecture & Assumptions

### 1. Camera & Mounting Configuration (`camera_calibration.json`)
VisionMCP is **100% self-contained** and stores all parameters in [`camera_calibration.json`](file:///mnt/c/Users/yosef/Documents/SCHOOL/Classes/2026%20Embodied%20Agentic%20AI/src/CleanupBot5/VisionMCP/camera_calibration.json).

> **You don't need to manually calculate or write most of these values!** Running the two calibration scripts automatically solves and saves them directly to `camera_calibration.json`:
> - **`calibrate_intrinsics.py`** automatically generates: `camera_matrix`, `dist_coeffs`, `focal_length_px`, and reference resolutions.
> - **`calibrate_extrinsics.py`** automatically generates: `camera_height_cm`, `pitch_bias_deg`, and `roll_bias_deg`.

#### What You Actually Need to Set:
* **`forward_offset_cm`** *(typically the only manual measurement)*: Horizontal distance along the robot heading from the center of the drive wheel axle to the camera lens (default: `10.0` cm). Calibration scripts detect the camera's pose relative to the floor marker, but only you know where your drive axle sits relative to the lens.
* **`joint_x_cm`, `joint_z_cm`** *(optional)*: Only needed if your camera tilts on an articulated arm joint (defaults: `8.0`, `7.0` cm).

#### Structure of `camera_calibration.json`:
```json
{
  "camera_height_cm": 11.91,     // [Auto-solved by calibrate_extrinsics] Floor to camera lens (cm)
  "pitch_bias_deg": 8.39,        // [Auto-solved by calibrate_extrinsics] Downward tilt angle (deg)
  "roll_bias_deg": -1.01,        // [Auto-solved by calibrate_extrinsics] Left/right roll tilt (deg)
  "forward_offset_cm": 10.0,     // [Set Manually] Axle to camera lens distance (cm)
  "focal_length_px": 954.17,     // [Auto-solved by calibrate_intrinsics] Focal length in pixels
  "reference_width_px": 1024,    // [Auto-solved by calibrate_intrinsics] Calibration resolution width
  "reference_height_px": 768,    // [Auto-solved by calibrate_intrinsics] Calibration resolution height
  "dist_coeffs": [...],          // [Auto-solved by calibrate_intrinsics] Lens distortion array
  "camera_matrix": [...]         // [Auto-solved by calibrate_intrinsics] 3x3 intrinsic matrix
}
```
*(Note: If you prefer not to run the ArUco calibration scripts at all, you can also rough-in `camera_height_cm` and `pitch_bias_deg` manually using a ruler and protractor).*

### 2. Camera & IMU Pitch
- **IMU Pitch**: The server subscribes to `/imu/data` (`sensor_msgs/msg/Imu`). Instantaneous robot/camera pitch directly adjusts the ground projection geometry in real time:
  $$\theta_{\text{total}} = \theta_{\text{pitch\_bias}} + \theta_{\text{imu\_pitch}}$$

### 3. Gripper Claws vs. Mobile Robots (Author's Differences)
My project (CleanupBot5) is configured for small floor object retrieval using mechanical parallel jaw grippers. If your robot differs:
- **Gripper Arm Visual Masking**:
  In [`vision_core/detector.py`](file:///mnt/c/Users/yosef/Documents/SCHOOL/Classes/2026%20Embodied%20Agentic%20AI/src/CleanupBot5/VisionMCP/vision_core/detector.py#L120-L123), detections touching the bottom 25 pixels in the central horizontal band ($0.25 \le x \le 0.75$) are filtered out so the robot's own claws are not detected as objects. If your robot does not have claws, you can disable or adjust this check if you need detection right at the bottom edge of the frame.
- **Object Size Limits**:
  In `vision_core/detector.py`, `min_grasp_width_cm = 1.0` and `max_grasp_width_cm = 4.5` define whether an object is flagged as `is_graspable: true`. If you are looking for larger objects (e.g., cups, boxes, bottles), adjust these bounds or read the raw `width_cm` and `shape` fields directly.
- **Grasp Verification (`verify_grasp`)**:
  This tool performs a visual macro-inspection on parallel jaw tips. If your robot does not have grippers, simply ignore this tool and use `scan_for_hazards()`.

### 4. Using TypeSafe Jev on Robots Without Grippers
Even without grippers, **TypeSafe Jev System One VLM** provides powerful visual reasoning for any mobile robot:
1. **Semantic Target Grounding (`locate_target`)**:
   When you call `scan_for_hazards(query="...")` with a specific item description (e.g. `"blue pill bottle"`, `"crushed red can"`, `"small metal key"`), Jev grounds and localizes the object on the floor even if YOLO's standard vocabulary misses it.
2. **Physical Orientation & Symmetry Inspection (`inspect_orientation`)**:
   Jev inspects target objects to determine their shape symmetry (`symmetric_round`, `elongated`, `irregular`), whether they rest diagonally/rotated, and computes the optimal approach angle (`head_on`, `approach_from_left`, `approach_from_right`).
3. **Custom Visual Property Audits**:
   Jev can easily be called on any cropped object to ask visual questions (e.g., container open vs. closed, full vs. empty, surface material, fragility, or hazard classification) for high-level agent planning.

---

## Connecting Your ESP32-CAM to ROS 2

VisionMCP subscribes to:
```
/camera/image_raw   (sensor_msgs/msg/Image, encoding: bgr8)
```

Most students in the class run the standard photo server firmware from Module 8 (`esp32cam_camera.ino`) on their AI-Thinker ESP32-CAM, which serves high-resolution photos over HTTP at:
```
http://<CAMERA_IP>/capture
```

To bridge this camera into ROS 2 so VisionMCP can process it, run this lightweight publisher node on your machine:

```python
#!/usr/bin/env python3
"""ROS 2 Bridge publishing ESP32-CAM /capture frames to /camera/image_raw."""

import urllib.request
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Replace with your ESP32-CAM's IP address
CAMERA_IP = "192.168.1.100"
CAPTURE_URL = f"http://{CAMERA_IP}/capture"

class ESP32CamNode(Node):
    def __init__(self):
        super().__init__("esp32_cam_bridge")
        self.publisher_ = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0 / 10.0, self.timer_callback) # 10 FPS
        self.get_logger().info(f"Streaming from ESP32-CAM: {CAPTURE_URL}")

    def timer_callback(self):
        try:
            req = urllib.request.Request(CAPTURE_URL, headers={"User-Agent": "ROS2-ESP32-Bridge"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                img_bytes = resp.read()
            frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "camera_optical_frame"
                self.publisher_.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to fetch frame from ESP32-CAM: {exc}")

def main():
    rclpy.init()
    node = ESP32CamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
```

---

## Minimal `pixi.toml` Configurations

The author's active environment uses **ROS 2 Jazzy**, but most students use **ROS 2 Humble**. Here are the minimal, clean `pixi.toml` templates without any unnecessary experimental packages (such as `transformers`, `einops`, or `timm`).

### Minimal `pixi.toml` for ROS 2 Humble (Recommended for Classmates)
```toml
[workspace]
name = "mcp-vision-server"
version = "0.1.0"
description = "FastMCP Vision Server (ROS 2 Humble)"
channels = ["https://prefix.dev/robostack-humble", "conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.10,<3.12"
opencv = ">=4.8"
ros-humble-rclpy = "*"
ros-humble-cv-bridge = "*"
ros-humble-sensor-msgs = "*"
ros-humble-nav-msgs = "*"

[pypi-dependencies]
mcp = "<2.0.0"
ultralytics = ">=8.3,<9"
openai-clip = ">=1.0"
typesafe-sdk = ">=0.7.5,<0.8"
python-dotenv = "*"

[tasks]
serve = "python mcp_vision_server.py"
```

### Minimal `pixi.toml` for ROS 2 Jazzy
```toml
[workspace]
name = "mcp-vision-server"
version = "0.1.0"
description = "FastMCP Vision Server (ROS 2 Jazzy)"
channels = ["https://prefix.dev/robostack-jazzy", "conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.11,<3.13"
opencv = ">=4.8"
ros-jazzy-rclpy = "*"
ros-jazzy-cv-bridge = "*"
ros-jazzy-sensor-msgs = "*"
ros-jazzy-nav-msgs = "*"

[pypi-dependencies]
mcp = "<2.0.0"
ultralytics = ">=8.3,<9"
openai-clip = ">=1.0"
typesafe-sdk = ">=0.7.5,<0.8"
python-dotenv = "*"

[tasks]
serve = "python mcp_vision_server.py"
```

> **Note on Weights**: Neural network weights (`yolov8s-worldv2.pt`) are **not** bundled in this repository. Ultralytics automatically downloads the required weights to `weights/` on the first execution.

---

## Repository Structure

```
VisionMCP/
├── mcp_vision_server.py       # Active FastMCP server (stdio JSON-RPC transport)
├── camera_calibration.json    # Camera intrinsics and floor-plane mounting extrinsics
├── pixi.toml / pixi.lock      # Pixi environment specification
├── .env                       # Environment variables (TYPESAFE_API_KEY optional)
│
├── vision_core/               # Perception library modules
│   ├── detector.py            # YOLO-World detection + CIELAB color + GroundProjector
│   ├── geometry.py            # Inverse perspective ground projection & kinematics
│   ├── color_analyzer.py      # CIELAB CSS3 color naming
│   ├── ros_bridge.py          # Thread-safe ROS 2 subscriber (/camera/image_raw, /imu/data, /odom)
│   └── verifier.py            # TypeSafe Jev visual reasoning & orientation inspection
│
└── tests/                     # Vision test & calibration tools
    ├── test_vision.py         # Live HUD video stream with 10cm perspective grid
    ├── calibrate_intrinsics.py # Camera matrix & distortion calibration
    └── calibrate_extrinsics.py # Ground-plane ArUco pitch & height calibration
```

---

## Exposed FastMCP Tools

### 1. `scan_for_hazards(query=None)`
Scans the floor for clutter and objects. Projects detections to real-world metric 2D coordinates $(X, Y)$ in meters.

- **Parameters**:
  - `query` *(optional string)*: Target object name or color (e.g. `"yellow cube"`, `"battery"`, `"bottle"`). If omitted or `None`, scans for all detected floor clutter.
- **Returns**:
  ```json
  {
    "hazards": [
      {
        "id": "hazard_1",
        "label": "yellow item",
        "color": "gold",
        "color_family": "yellow",
        "is_multicolored": false,
        "world_x_m": 0.42,
        "world_y_m": -0.05,
        "standoff_pose": {
          "x_m": 0.12,
          "y_m": -0.02,
          "yaw_deg": -6.8
        },
        "shape": "cube",
        "width_cm": 2.8,
        "is_diagonal": false,
        "needs_extended_clamp": false,
        "approach": "head_on",
        "shape_symmetry": "symmetric_round",
        "is_graspable": true
      }
    ],
    "robot_pose": {
      "x_m": 0.0,
      "y_m": 0.0,
      "yaw_deg": 0.0
    },
    "summary": "Detected 1 hazard(s) in world frame: #hazard_1 'yellow item' at world (X=0.42m, Y=-0.05m), standoff: (X=0.12m, Y=-0.02m, theta=-6.8 deg), width: 2.8cm, diagonal: False, approach: head_on"
  }
  ```

### 2. `verify_grasp(phase="post_lift")`
*(Optional — for gripper-equipped robots with `TYPESAFE_API_KEY`)*
Verifies whether the robot gripper successfully clamped an object.

- **Parameters**:
  - `phase`: `"pre_lift"` (audits closed jaws on floor) or `"post_lift"` (audits jaws raised in the air).
- **Returns**:
  ```json
  {
    "grasped": true,
    "outcome": "SUCCESS",
    "confidence": 0.94,
    "note": "TypeSafe Jev post_lift check: state=holding_object, holding=True (P=0.94)"
  }
  ```

---

## Quick Start Guide

### 1. Environment Setup (Pixi)
Ensure [Pixi](https://pixi.sh) is installed, then set up the environment:
```bash
cd VisionMCP
pixi install
```

### 2. Start the FastMCP Server (stdio transport)
```bash
cd VisionMCP
pixi run serve
# Or directly:
python mcp_vision_server.py
```
Add to your agent's MCP configuration:
```json
{
  "mcpServers": {
    "vision": {
      "command": "pixi",
      "args": ["run", "serve"],
      "cwd": "VisionMCP"
    }
  }
}
```

### 3. Camera Calibration
Calibrating your camera writes the optical and physical parameters directly to `camera_calibration.json`.

> **Marker Width Configuration**: Before running calibration, measure your printed 4x4 ArUco marker's outer black square width (in cm) and update `MARKER_SIZE_CM` at the top of both calibration files ([`tests/calibrate_intrinsics.py`](file:///mnt/c/Users/yosef/Documents/SCHOOL/Classes/2026%20Embodied%20Agentic%20AI/src/CleanupBot5/VisionMCP/tests/calibrate_intrinsics.py#L48) and [`tests/calibrate_extrinsics.py`](file:///mnt/c/Users/yosef/Documents/SCHOOL/Classes/2026%20Embodied%20Agentic%20AI/src/CleanupBot5/VisionMCP/tests/calibrate_extrinsics.py#L52), default: `10.6` cm). You can also pass `--marker-size <cm>` as a command-line argument.

- **Intrinsics Calibration (`calibrate_intrinsics.py`)**:
  - **What it does**: Solves for internal optical properties—focal length ($f_x, f_y$), optical center ($c_x, c_y$), and lens distortion coefficients. This corrects for camera lens curvature (barrel/pincushion distortion) so pixel rays project into 3D space with accurate perspective.
  - **How to run**: Hold an ArUco marker sheet (or checkerboard) at various distances and tilt angles across the camera frame:
    ```bash
    pixi run python tests/calibrate_intrinsics.py --live
    ```
    *(Press `SPACE` to capture ~15–25 frames, `c` to compute, and `s` to save to `camera_calibration.json`).*

- **Extrinsics Calibration (`calibrate_extrinsics.py`)**:
  - **What it does**: Solves for the camera's physical 3D mounting pose relative to the floor plane and drive wheel axle—specifically camera height above ground (`camera_height_cm`) and downward tilt angle (`pitch_bias_deg`). This enables accurate inverse perspective mapping from image pixels to real-world ground coordinates $(X, Y)$ in centimeters and meters.
  - **How to run**: Place one or multiple standard ArUco markers (e.g. 2 markers using IDs 0 and 1) flat on the floor in front of the robot. Using multiple markers allows the solver to cross-validate and compute a more accurate ground-plane consensus:
    ```bash
    pixi run python tests/calibrate_extrinsics.py --live
    ```
    *(Press `s` to save the measured height and pitch offset directly to `camera_calibration.json`).*

### 4. Live HUD Visualizer & Perspective Grid Test
Once calibrated, run the HUD visualizer to verify that your calibration is accurate. It displays the live camera stream overlaid with a 10 cm ground-plane perspective grid, detected YOLO bounding boxes, CIELAB color labels, and estimated distances:
```bash
pixi run python tests/test_vision.py
```
- **Hotkeys**:
  - `j`: Trigger TypeSafe Jev audit.
  - `o`: Trigger TypeSafe Jev orientation analysis on target.
  - `s`: Save live annotated snapshot to `VisionMCP/images/`.
  - `q`: Exit.
