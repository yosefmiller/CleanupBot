"""Standalone MCP server exposing offline robot datasheet lookup tools.

Normally this is launched as a subprocess by rosa_agent_mcp.py over stdio.
"""
from mcp.server import FastMCP

mcp = FastMCP("robot-datasheets")

BNO055_NOTES = {
    "calibration": "BNO055 calibration status ranges 0-3 per sensor (system, gyro, accel, mag). "
                   "Read via the CALIB_STAT register (0x35); 3 = fully calibrated.",
    "i2c address": "Default I2C address is 0x28 (0x29 if the ADR pin is pulled high).",
    "output modes": "Supports NDOF fusion mode (0x0C) for absolute orientation using accel+gyro+mag.",
}

TB6612FNG_NOTES = {
    "pinout": "AIN1/AIN2/PWMA control Motor A; BIN1/BIN2/PWMB control Motor B; STBY must be high to enable outputs.",
    "voltage": "VM (motor supply) up to 15V; VCC (logic) 2.7-5.5V; logic pins are 3.3V-safe for the Pico.",
    "direction control": "Set AIN1=1, AIN2=0 for forward; AIN1=0, AIN2=1 for reverse; PWM pin sets speed.",
}

GY_BNO085_NOTES = {
    "i2c address": "Default I2C address is 0x4A. It becomes 0x4B if the ADDR (or MOSI) pin is pulled high.",
    "protocol selection": "PS0 and PS1 pins select the interface: 0,0 for I2C (default); 0,1 for UART-RVC; 1,0 for UART-SHTP; 1,1 for SPI.",
    "accuracy": "Dynamic rotation error: 3.5° (2.5° gaming). Heading drift: 0.5°/min. Accelerometer: 0.3m/s². Gyroscope: 3.1°/sec.",
    "pins and power": "Input voltage is 3.3V-5V. Standard pins: VCC, GND, SCL/RX, SDA/TX, ADDR/MOSI, CS, INT, RST, PS1, PS0."
}

ESP32S3_CAM_NOTES = {
    "specs and features": "Uses ESP32-S3R8 chip with 8MB PSRAM and 8MB Flash. Features native Type-C USB (no external serial adapter needed) and an onboard 3.7V battery charging circuit.",
    "board pinout": "Left Pins: RES, 44, 43, 42, 41, 46, GND, 3V3. Right Pins: 0, 14, 47, NC, 48, 1, GND, 5V.",
    "camera wiring": "OV3660/OV2640 pinout defaults: Data D0-D7 mapped to GPIOs 11, 9, 8, 10, 12, 18, 17, 16. VSYNC: 6, HREF: 7, PCLK: 13, XCLK: 15. SDA: 4, SCL: 5."
}

TT_ENCODER_MOTOR_NOTES = {
    "gearbox and torque": "NULLLAB 1:90 reduction ratio with an all-metal gearbox. Rated torque: 1.0 kg*cm; Stall torque: 1.8 kg*cm.",
    "speed and power": "Motor operates at DC 3-12V. No-load output shaft speed is ~113 RPM (internal motor spins at 10,000 RPM before the 1:90 reduction).",
    "encoder specs": "12 PPR AB-Phase Hall Encoder requiring 3.3-5V. Factoring in the 1:90 gear ratio, the output shaft yields exactly 1,080 pulses per full revolution (12 * 90)."
}

def search_docs(notes_dict: dict, query: str) -> str:
    """Generic search function for datasheet notes."""
    query_lower = query.lower()
    for key, note in notes_dict.items():
        if key in query_lower or query_lower in key:
            return note
    return f"No note found for '{query}'. Available topics: {', '.join(notes_dict)}"

@mcp.tool()
def search_bno055_docs(query: str) -> str:
    """Search offline BNO055 IMU datasheet notes for a keyword."""
    return search_docs(BNO055_NOTES, query)

@mcp.tool()
def search_tb6612fng_docs(query: str) -> str:
    """Search offline TB6612FNG motor driver datasheet notes for a keyword."""
    return search_docs(TB6612FNG_NOTES, query)

@mcp.tool()
def search_bno085_docs(query: str) -> str:
    """Search offline BNO085 IMU datasheet notes for a keyword."""
    return search_docs(GY_BNO085_NOTES, query)

@mcp.tool()
def search_esp32s3_cam_docs(query: str) -> str:
    """Search offline ESP32-S3-CAM datasheet notes for a keyword."""
    return search_docs(ESP32S3_CAM_NOTES, query)

@mcp.tool()
def search_tt_encoder_motor_docs(query: str) -> str:
    """Search offline TT encoder motor datasheet notes for a keyword."""
    return search_docs(TT_ENCODER_MOTOR_NOTES, query)

@mcp.tool()
def list_available_docs() -> dict:
    """List available datasheet notes for the robot's components."""
    return {
        "BNO055 IMU": list(BNO055_NOTES.keys()),
        "TB6612FNG Motor Driver": list(TB6612FNG_NOTES.keys()),
        "BNO085 IMU": list(GY_BNO085_NOTES.keys()),
        "ESP32-S3-CAM": list(ESP32S3_CAM_NOTES.keys()),
        "TT Encoder Motor": list(TT_ENCODER_MOTOR_NOTES.keys()),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")