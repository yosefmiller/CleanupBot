"""MicroPython MJPEG streaming server for nulllaborg ESP32-S3 CAM
using cnadler86/micropython-camera-API (v0.6.0).
"""

import gc
import time
import network
import usocket as socket
from machine import Pin
from camera import Camera, FrameSize, PixelFormat, GrabMode

time.sleep(1)
print("--- Starting ESP32-S3 CAM ---")

# ── 1. Wi-Fi Setup & Connection Indicator ───────────────────────────
def load_env(filepath=".env"):
    """Load key-value pairs from a .env file for MicroPython."""
    env = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
    return env

_env = load_env(".env")
WIFI_SSID = _env.get("WIFI_SSID", "")
WIFI_PASSWORD = _env.get("WIFI_PASSWORD", "")
STATIC_IP = _env.get("STATIC_IP", "192.168.137.150")
NETMASK   = _env.get("NETMASK", "255.255.255.0")
GATEWAY   = _env.get("GATEWAY", "192.168.137.1")
DNS       = _env.get("DNS", "192.168.137.1")

if not WIFI_SSID:
    print("ERROR: WIFI_SSID not found in .env! Please provide credentials in .env file.")

# Onboard LED indicator (GPIO 3)
led = Pin(3, Pin.OUT)
led.value(0)

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.ifconfig((STATIC_IP, NETMASK, GATEWAY, DNS))


def connect_wifi():
    """Wait for WIFI_SSID (slow flash), then connect (rapid flash during handshake)."""
    while not wlan.isconnected():
        print(f"Scanning for Wi-Fi SSID '{WIFI_SSID}'...")
        ssid_found = False
        try:
            for net in wlan.scan():
                if net[0].decode("utf-8", "ignore") == WIFI_SSID:
                    ssid_found = True
                    break
        except Exception as e:
            print("Scan error:", e)

        if not ssid_found:
            # Slow flash while waiting for SSID (500ms ON / 500ms OFF)
            led.value(1)
            time.sleep_ms(500)
            led.value(0)
            time.sleep_ms(500)
            continue

        # SSID found: initiate handshake
        print(f"SSID '{WIFI_SSID}' found. Connecting...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)

        # Rapid flash during handshake (100ms ON / 100ms OFF, up to 10s)
        for _ in range(50):
            if wlan.isconnected():
                break
            led.value(1)
            time.sleep_ms(100)
            led.value(0)
            time.sleep_ms(100)

        if wlan.isconnected():
            break
        print("Handshake timed out. Retrying...")

    # Connected: 500ms solid confirmation pulse then off
    led.value(1)
    time.sleep_ms(500)
    led.value(0)
    print(f"Connected! IP: {wlan.ifconfig()[0]}")


# Establish initial Wi-Fi connection
connect_wifi()

# ── 2. Camera Setup (Optimized for ROS 2 & YOLO) ────────────────────
# - FrameSize.VGA (640x480): 1:1 match for YOLOv8 input, low latency, low PSRAM
# - PixelFormat.JPEG: Hardware compressed on-chip, ~30KB/frame
# - xclk_freq 10MHz: Stable DMA clocking on nulllaborg hardware
# - fb_count 2 + GrabMode.LATEST: Real-time observation, zero buffer queue lag
# - TARGET_FPS (10): Deliberately throttled from unconstrained ~20 FPS to 10 FPS
#   to cut Wi-Fi bandwidth by ~50%, reduce heat, and drastically extend robot battery life.
TARGET_FPS = 10
FRAME_INTERVAL_MS = int(1000 / TARGET_FPS)

cam = Camera(
    pixel_format=PixelFormat.JPEG,
    frame_size=FrameSize.VGA,
    jpeg_quality=85,
    xclk_freq=10000000,
    fb_count=2,
    grab_mode=GrabMode.LATEST,
)

# White balance (Office WB)
cam.whitebal = True
cam.wb_mode = 3

# Verified robot mount orientation:
cam.vflip = False
cam.hmirror = True

# Prime DMA buffers to clear initial frames
for _ in range(2):
    cam.capture()

print(f"Camera ready: vflip={cam.vflip}, hmirror={cam.hmirror}")

# ── 3. Streaming Server ─────────────────────────────────────────────
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 81))
server.listen(1)
server.settimeout(2.0)  # Non-blocking accept allows checking Wi-Fi status

ip_address = wlan.ifconfig()[0]
print(f"Stream live at: http://{ip_address}:81/stream")

BOUNDARY = "123456789000000000000987654321"

while True:
    # Auto-reconnect if Wi-Fi dropped during operation
    if not wlan.isconnected():
        print("Wi-Fi disconnected! Reconnecting...")
        connect_wifi()

    try:
        client, addr = server.accept()
    except OSError:
        # Accept timed out; loop to check Wi-Fi health
        continue

    # Set timeout on client to prevent blocking forever on dead connections
    client.settimeout(3.0)
    try:
        request = client.recv(1024)
        if not request:
            client.close()
            continue

        # Stream MJPEG
        client.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: multipart/x-mixed-replace; boundary="
            + BOUNDARY.encode()
            + b"\r\n\r\n"
        )

        frame_count = 0
        drop_count = 0

        while True:
            t_start = time.ticks_ms()
            frame = cam.capture()
            if not frame:
                drop_count += 1
                time.sleep_ms(5)
                # Break out and reset connection if camera DMA stalls
                if drop_count > 20:
                    print("Camera capture dropped 20 frames; resetting stream.")
                    break
                continue
            drop_count = 0

            header = (
                f"--{BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n\r\n"
            ).encode()

            client.write(header)
            client.write(frame)
            client.write(b"\r\n")

            # Periodic garbage collection to prevent heap fragmentation
            frame_count += 1
            if frame_count % 30 == 0:
                gc.collect()

            # Paced FPS: sleeps remaining interval to reduce CPU load, heat, and Wi-Fi battery drain
            elapsed_ms = time.ticks_diff(time.ticks_ms(), t_start)
            sleep_ms = FRAME_INTERVAL_MS - elapsed_ms
            if sleep_ms > 0:
                time.sleep_ms(sleep_ms)
            else:
                time.sleep_ms(1)  # Minimal yield for lwIP TCP/IP stack

    except OSError:
        # Socket timeout or client disconnected
        pass
    finally:
        client.close()
