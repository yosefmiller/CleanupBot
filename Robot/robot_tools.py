from langchain.agents import tool


@tool
def read_imu() -> dict:
    """Read the BNO055 IMU sensor and return orientation data.

    Returns a dictionary with heading (yaw), roll, and pitch in degrees.
    Use this tool when you need to know the robot's current orientation.
    """
    # In production, this calls a ROS 2 service or reads from a topic.
    # For demonstration, we return simulated data.
    return {
        "heading": 47.3,
        "roll": 0.1,
        "pitch": -0.3,
        "calibration_status": "fully_calibrated"
    }


@tool
def move_forward(distance: float) -> str:
    """Move the robot forward by the specified distance in meters.

    :param distance: Distance to move forward in meters (0.1 to 5.0).

    Publishes a Twist message to /cmd_vel. Returns a success or error message.
    Use this when the user asks the robot to move forward.
    """
    if distance < 0.1 or distance > 5.0:
        return f"Error: distance {distance}m is out of safe range (0.1-5.0m)."

    # In production, publish to /cmd_vel and wait for odometry feedback.
    return f"Moved forward {distance}m successfully."


@tool
def read_tof_distance() -> dict:
    """Read the VL53L0X Time-of-Flight distance sensor.

    Returns the distance to the nearest obstacle in meters.
    Use this to check for obstacles before moving.
    """
    return {"distance_m": 1.45, "status": "valid"}


@tool
def rotate(angle_degrees: float) -> str:
    """Rotate the robot by the specified angle in degrees.

    :param angle_degrees: Positive = counter-clockwise, negative = clockwise.

    Publishes angular velocity to /cmd_vel and monitors IMU for completion.
    """
    return f"Rotated {angle_degrees} degrees."


@tool
def emergency_stop() -> str:
    """Immediately stop all robot movement.

    Publishes zero velocity to /cmd_vel and /estop.
    Use this if something is wrong or an obstacle is too close.
    """
    return "Emergency stop executed. All motors halted."


@tool
def get_robot_status() -> dict:
    """Get the current status of the robot including position, battery, and sensor health.

    Returns a dictionary with odometry position, battery percentage,
    and sensor connectivity status.
    """
    return {
        "position": {"x": 2.1, "y": 0.8, "theta": 47.3},
        "battery_pct": 87,
        "sensors": {"imu": "ok", "tof": "ok", "encoders": "ok"}
    }