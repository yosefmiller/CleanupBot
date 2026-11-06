# ESP32-S3 CAM MicroPython Setup

Streaming firmware and server for the **nulllaborg ESP32-S3 CAM** development board (ESP32-S3R8 SoC with 8MB Flash & 8MB Octal PSRAM).

Firmware: **`cnadler86/micropython-camera-API` (Release `v0.6.0` on MicroPython v1.27.0)**.

---

## 1. Quick Start / Flashing

### Hardware Pinout
The Freenove ESP32S3 CAM build targets the exact pinout of the nulllaborg board:
* `XCLK`: GPIO 15
* `SIOD (SDA)`: GPIO 4, `SIOC (SCL)`: GPIO 5
* `D0..D7`: GPIO 11, 9, 8, 10, 12, 18, 17, 16
* `VSYNC`: GPIO 6, `HREF`: GPIO 7, `PCLK`: GPIO 13

### Flashing (PowerShell with `uv`)

1. **Download and Extract Firmware (v0.6.0):**
   ```powershell
   # Create firmware folder (if not present)
   New-Item -ItemType Directory -Force -Path esp32s3_cam\firmware

   # Download release zip
   curl.exe -L -o esp32s3_cam\firmware\firmware.zip https://github.com/cnadler86/micropython-camera-API/releases/download/v0.6.0/mpy_cam-v1.27.0-FREENOVE_ESP32S3_CAM.zip

   # Extract firmware.bin
   tar.exe -xf esp32s3_cam\firmware\firmware.zip -C esp32s3_cam\firmware\
   Remove-Item esp32s3_cam\firmware\firmware.zip
   ```
   *(Alternatively in pure PowerShell: `Expand-Archive -Path esp32s3_cam\firmware\firmware.zip -DestinationPath esp32s3_cam\firmware\ -Force`)*

2. Put the board into bootloader mode (**hold BOOT, tap RST, release BOOT**).
3. Erase and write flash:
   ```powershell
   uvx esptool --port COM7 erase-flash
   uvx esptool --port COM7 --chip esp32s3 --baud 460800 write-flash --flash-mode dio --flash-size 8MB 0x0 esp32s3_cam\firmware\firmware.bin
   ```
4. Press the **RST** button to reboot the board.

---

## 2. Upload Code to Board

Upload `.env` and `main.py` directly to the ESP32 using `mpremote`:

```powershell
uvx mpremote connect COM7 cp esp32s3_cam\.env :.env
uvx mpremote connect COM7 cp esp32s3_cam\main.py :main.py
uvx mpremote connect COM7 reset
```

To view live boot and Wi-Fi logs in the REPL:
```powershell
uvx mpremote connect COM7 repl
```

---

## 3. Video Stream & Port Proxy

Once connected to Wi-Fi at `192.168.137.150:81`, proxy the port from the Windows Mobile Hotspot network to `localhost`:

```powershell
netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=8081 connectaddress=192.168.137.150 connectport=81
```

* **Live MJPEG Stream (Browser / ROS 2 / OpenCV):**
  ```text
  http://127.0.0.1:8081/stream
  ```
  *(Hardware orientation is locked to `FLIP = 0, MIRROR = 1` for robot mounting).*

---

## 4. ROS 2 Camera Node (`esp32_cam_node.py`)

Publishes frames to `/camera/image_raw` and `/camera/image_raw/compressed`. Defaults to `http://127.0.0.1:8081/stream`.

```bash
# Standard stream (uses default http://127.0.0.1:8081/stream)
python3 esp32s3_cam/esp32_cam_node.py

# Explicit URL
python3 esp32s3_cam/esp32_cam_node.py http://127.0.0.1:8081/stream

# With host-side OpenCV flip (0=vertical, 1=horizontal, -1=both/180°)
python3 esp32s3_cam/esp32_cam_node.py --flip 0
python3 esp32s3_cam/esp32_cam_node.py --flip -1
```