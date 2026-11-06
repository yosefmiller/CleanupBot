#!/usr/bin/env python3
"""
imu_live_visualizer.py
----------------------
Live 3D Web Visualizer & Terminal Monitor for CleanupBot Pico 2 W IMU.

Subscribes to:
  - /pico/imu/pitch  (std_msgs/msg/Float32)
  - /pico/tilt/status (std_msgs/msg/Int32)

Features:
  1. Serves an interactive 3D Web Dashboard on http://localhost:8080.
     Open it in Chrome/Edge on your Windows host or inside WSL to watch the robot
     tilt and rotate in 3D in real time!
  2. Displays an ASCII gauge in the terminal with live pitch and reflex state.

Usage:
  pixi run python src/Robot/imu_live_visualizer.py
"""

import json
import math
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Int32

# Global Telemetry State shared between ROS 2 subscriber and Web Server
telemetry_lock = threading.Lock()
telemetry_data = {
    "pitch": 0.0,
    "status_code": 0,
    "status_label": "IDLE",
    "last_update": 0,
    "message_count": 0,
}

STATUS_NAMES = {
    0: "IDLE (Standing by)",
    1: "ARMED_FORWARD (Waiting for forward tilt)",
    2: "ARMED_BACKWARD (Waiting for backward tilt)",
    3: "RUNNING_FORWARD (Wheels driving forward 10s)",
    4: "RUNNING_BACKWARD (Wheels driving backward 10s)",
    5: "STOPPED_FORWARD (Move complete; stopped)",
    6: "STOPPED_BACKWARD (Move complete; stopped)",
    7: "ERROR (IMU timeout or read error)",
    8: "BUSY (Already armed/running)",
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CleanupBot Live 3D IMU Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; }
  </style>
</head>
<body class="p-6">
  <div class="max-w-4xl mx-auto space-y-6">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-wrap items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">Live ROS 2 Telemetry</span>
          <span id="connBadge" class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Streaming</span>
        </div>
        <h1 class="text-2xl font-bold mt-2 text-white">CleanupBot Real-Time 3D Orientation</h1>
        <p class="text-slate-400 text-sm mt-1">Listening to <code class="text-indigo-300">/pico/imu/pitch</code> and <code class="text-indigo-300">/pico/tilt/status</code></p>
      </div>
      <div class="text-right">
        <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Live Robot Pitch</div>
        <div id="pitchBig" class="text-4xl font-extrabold text-indigo-400">0.0&deg;</div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- 3D Canvas -->
      <div class="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col items-center shadow-xl">
        <canvas id="canvas3d" width="560" height="380" class="w-full max-w-[560px] h-[380px] bg-slate-950 rounded-xl border border-slate-800"></canvas>
        <div class="w-full grid grid-cols-2 gap-3 mt-4 text-xs">
          <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/50">
            <span class="font-bold text-slate-200">IMU Axes:</span>
            <div class="text-rose-400 font-semibold mt-0.5">&bull; X_imu: Points RIGHT (Wheel Axle)</div>
            <div class="text-emerald-400 font-semibold">&bull; Y_imu: Points BACKWARDS</div>
            <div class="text-sky-400 font-semibold">&bull; Z_imu: Points UP</div>
          </div>
          <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/50">
            <span class="font-bold text-slate-200">Reflex Trigger Rule:</span>
            <div class="text-slate-300 mt-0.5">&bull; Forward Tilt: Pitch &ge; +2.0&deg; (Hold 100ms)</div>
            <div class="text-slate-300">&bull; Backward Tilt: Pitch &le; -2.0&deg; (Hold 100ms)</div>
            <div class="text-slate-300">&bull; Motors run for 10.0 seconds then auto-stop</div>
          </div>
        </div>
      </div>

      <!-- Status sidebar -->
      <div class="lg:col-span-4 space-y-6">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div class="text-xs font-semibold uppercase tracking-wider text-slate-400">Reflex State Machine</div>
          <div id="stateCard" class="p-3.5 rounded-xl border border-slate-700 bg-slate-800 flex items-center justify-between">
            <div>
              <div class="text-xs text-slate-400 uppercase">Current State</div>
              <div id="stateText" class="text-sm font-bold text-slate-200">IDLE</div>
            </div>
            <div id="stateDot" class="w-3.5 h-3.5 rounded-full bg-slate-500"></div>
          </div>

          <div>
            <div class="text-xs text-slate-400 mb-1 flex justify-between">
              <span>-2.0&deg;</span><span>Tilt Gauge</span><span>+2.0&deg;</span>
            </div>
            <div class="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
              <div class="absolute left-0 top-0 bottom-0 w-[40%] bg-rose-500/30"></div>
              <div class="absolute right-0 top-0 bottom-0 w-[40%] bg-emerald-500/30"></div>
              <div id="gaugeMarker" class="absolute top-0 bottom-0 w-2 -ml-1 bg-white rounded-full shadow" style="left: 50%;"></div>
            </div>
          </div>

          <div class="pt-3 border-t border-slate-800 text-xs space-y-1 text-slate-400 font-mono">
            <div>Total Updates: <span id="msgCount" class="text-slate-200 font-bold">0</span></div>
            <div>Last Packet: <span id="lastTime" class="text-slate-200">Just now</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('canvas3d');
    const ctx = canvas.getContext('2d');
    const pitchBig = document.getElementById('pitchBig');
    const stateText = document.getElementById('stateText');
    const stateCard = document.getElementById('stateCard');
    const stateDot = document.getElementById('stateDot');
    const gaugeMarker = document.getElementById('gaugeMarker');
    const msgCount = document.getElementById('msgCount');
    const lastTime = document.getElementById('lastTime');

    let currentPitch = 0.0;

    function pollTelemetry() {
      fetch('/telemetry')
        .then(r => r.json())
        .then(data => {
          currentPitch = data.pitch;
          pitchBig.textContent = (currentPitch >= 0 ? '+' : '') + currentPitch.toFixed(2) + '°';
          stateText.textContent = data.status_label;
          msgCount.textContent = data.message_count;
          lastTime.textContent = (Date.now() / 1000 - data.last_update).toFixed(1) + 's ago';

          const pct = Math.max(0, Math.min(100, ((currentPitch + 25) / 50) * 100));
          gaugeMarker.style.left = pct + '%';

          if (data.status_code === 3 || data.status_code === 4) {
            stateCard.className = 'p-3.5 rounded-xl border border-emerald-500/50 bg-emerald-500/15 flex items-center justify-between';
            stateDot.className = 'w-3.5 h-3.5 rounded-full bg-emerald-400 animate-pulse';
          } else if (data.status_code === 1 || data.status_code === 2) {
            stateCard.className = 'p-3.5 rounded-xl border border-amber-500/50 bg-amber-500/15 flex items-center justify-between';
            stateDot.className = 'w-3.5 h-3.5 rounded-full bg-amber-400 animate-ping';
          } else {
            stateCard.className = 'p-3.5 rounded-xl border border-slate-700 bg-slate-800 flex items-center justify-between';
            stateDot.className = 'w-3.5 h-3.5 rounded-full bg-slate-500';
          }

          draw3D(currentPitch);
        })
        .catch(err => console.error(err));
    }

    setInterval(pollTelemetry, 60);

    // Interactive Mouse Orbit Controls
    let camYaw = -0.65;    // Azimuth perspective
    let camPitch = 0.35;   // Elevation perspective
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;

    canvas.style.cursor = 'grab';
    canvas.addEventListener('mousedown', e => {
      isDragging = true;
      canvas.style.cursor = 'grabbing';
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    });
    window.addEventListener('mousemove', e => {
      if (!isDragging) return;
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      camYaw += dx * 0.01;
      camPitch = Math.max(-1.1, Math.min(1.1, camPitch + dy * 0.01));
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      draw3D(currentPitch);
    });
    window.addEventListener('mouseup', () => {
      isDragging = false;
      canvas.style.cursor = 'grab';
    });
    canvas.addEventListener('dblclick', () => {
      camYaw = -0.65;
      camPitch = 0.35;
      draw3D(currentPitch);
    });

    function draw3D(pitchDeg) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cx = canvas.width / 2;
      const cy = canvas.height / 2 + 30;
      const p = (pitchDeg * Math.PI) / 180.0;

      function project(x, y, z) {
        // 1. Physical Robot Pitch:
        //    Pivots around the lateral wheel axle (y axis in the 3D model).
        //    When pitch is positive (nose down): Front (+x) tilts down (-z), Rear (-x) tilts up (+z)
        let x1 = x * Math.cos(p) + z * Math.sin(p);
        let y1 = y;
        let z1 = -x * Math.sin(p) + z * Math.cos(p);

        // 2. Camera View Orbit (Yaw & Pitch):
        let x2 = x1 * Math.cos(camYaw) - y1 * Math.sin(camYaw);
        let y2 = x1 * Math.sin(camYaw) + y1 * Math.cos(camYaw);
        let z2 = z1;

        let x3 = x2;
        let y3 = y2 * Math.cos(camPitch) - z2 * Math.sin(camPitch);
        let z3 = y2 * Math.sin(camPitch) + z2 * Math.cos(camPitch);

        const sx = cx + x3 * 1.35;
        const sy = cy - z3 * 1.35;
        return { x: sx, y: sy };
      }

      // Grid on table
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let g = -150; g <= 150; g += 50) {
        let p1 = project(g, -150, -40);
        let p2 = project(g, 150, -40);
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        let p3 = project(-150, g, -40);
        let p4 = project(150, g, -40);
        ctx.beginPath(); ctx.moveTo(p3.x, p3.y); ctx.lineTo(p4.x, p4.y); ctx.stroke();
      }

      const L = 80, W = 60, H = 20;
      const v = [
        project( L, -W, -H), project( L,  W, -H), project(-L,  W, -H), project(-L, -W, -H),
        project( L, -W,  H), project( L,  W,  H), project(-L,  W,  H), project(-L, -W,  H),
      ];

      function poly(indices, fill, stroke) {
        ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(v[indices[0]].x, v[indices[0]].y);
        for (let i = 1; i < indices.length; i++) ctx.lineTo(v[indices[i]].x, v[indices[i]].y);
        ctx.closePath(); ctx.fill(); ctx.stroke();
      }

      poly([0,1,2,3], '#0f172a', '#1e293b');
      poly([1,2,6,5], '#1e293b', '#334155');
      poly([2,3,7,6], '#1e293b', '#334155');
      poly([0,3,7,4], '#334155', '#475569');
      poly([0,1,5,4], '#3b82f6', '#60a5fa'); // Front bumper in blue
      poly([4,5,6,7], '#1e293b', '#475569');

      // Draw Wheels on Left (y=+W) and Right (y=-W)
      function drawWheel(yCenter) {
        ctx.fillStyle = '#0f172a';
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 2;
        const wRadius = 26;
        const pts = [];
        for (let deg = 0; deg <= 360; deg += 30) {
          let rad = (deg * Math.PI) / 180;
          let wx = Math.cos(rad) * wRadius;
          let wz = -H + Math.sin(rad) * wRadius;
          pts.push(project(wx, yCenter, wz));
        }
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      drawWheel( W + 6);
      drawWheel(-W - 6);

      // Front Gripper
      const g1 = project(L, -20, 0), g2 = project(L + 35, -25, 0), g3 = project(L + 35, 25, 0), g4 = project(L, 20, 0);
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(g1.x, g1.y); ctx.lineTo(g2.x, g2.y); ctx.moveTo(g4.x, g4.y); ctx.lineTo(g3.x, g3.y); ctx.stroke();

      // Front label
      const fl = project(L + 20, 0, H + 12);
      ctx.fillStyle = '#f59e0b'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('FRONT (Gripper)', fl.x, fl.y);

      // IMU box
      const imuZ = H + 2;
      const ib = [project(24, -18, imuZ), project(24, 18, imuZ), project(-24, 18, imuZ), project(-24, -18, imuZ)];
      ctx.fillStyle = '#0284c7'; ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ib[0].x, ib[0].y);
      for (let i = 1; i < ib.length; i++) ctx.lineTo(ib[i].x, ib[i].y);
      ctx.closePath(); ctx.fill(); ctx.stroke();

      // IMU Coordinate Vectors
      const origin = project(0, 0, imuZ);
      function vec(dest, color, label) {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(dest.x, dest.y); ctx.stroke();
        ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
        ctx.fillText(label, dest.x + 6, dest.y + 4);
      }
      vec(project(0, -60, imuZ), '#f43f5e', 'X_imu (Right)');
      vec(project(-65, 0, imuZ), '#10b981', 'Y_imu (Back)');
      vec(project(0, 0, imuZ + 55), '#38bdf8', 'Z_imu (Up)');

      // Hint text on canvas
      ctx.fillStyle = '#64748b';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Click & drag to rotate 3D view (double-click to reset)', 12, canvas.height - 12);
    }
  </script>
</body>
</html>
"""


class TelemetryHttpHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with telemetry_lock:
                data = json.dumps(telemetry_data)
            self.wfile.write(data.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP access logs
        return


class ImuVisualizerNode(Node):
    def __init__(self):
        super().__init__("imu_live_visualizer")

        # Use best-effort QoS matching the Pico 2 W publisher
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pitch_sub = self.create_subscription(
            Float32, "/pico/imu/pitch", self.pitch_callback, qos_best_effort
        )
        self.status_sub = self.create_subscription(
            Int32, "/pico/tilt/status", self.status_callback, qos_best_effort
        )

        self.get_logger().info("IMU Live Visualizer Node started.")
        self.get_logger().info("Subscribed to /pico/imu/pitch and /pico/tilt/status")

    def pitch_callback(self, msg: Float32):
        with telemetry_lock:
            telemetry_data["pitch"] = round(float(msg.data), 2)
            telemetry_data["last_update"] = time.time()
            telemetry_data["message_count"] += 1

    def status_callback(self, msg: Int32):
        code = int(msg.data)
        with telemetry_lock:
            telemetry_data["status_code"] = code
            telemetry_data["status_label"] = STATUS_NAMES.get(code, f"UNKNOWN({code})")
            telemetry_data["last_update"] = time.time()


def start_web_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), TelemetryHttpHandler)
    server.serve_forever()


def main():
    rclpy.init()
    node = ImuVisualizerNode()

    # Start HTTP Web Server in background daemon thread
    web_thread = threading.Thread(target=start_web_server, args=(8080,), daemon=True)
    web_thread.start()

    print("\n" + "=" * 65)
    print(" 🚀 CLEANUPBOT LIVE 3D IMU VISUALIZER ACTIVE")
    print("=" * 65)
    print(" Open in your browser:   http://localhost:8080")
    print(" Streaming from topics:  /pico/imu/pitch & /pico/tilt/status")
    print(" Press Ctrl+C in terminal to exit.")
    print("=" * 65 + "\n")

    # Spin ROS 2 in background thread
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()

    try:
        while True:
            with telemetry_lock:
                pitch = telemetry_data["pitch"]
                label = telemetry_data["status_label"]
                count = telemetry_data["message_count"]

            # ASCII tilt bar (-25 deg to +25 deg)
            bar_len = 40
            pos = int(((pitch + 25.0) / 50.0) * bar_len)
            pos = max(0, min(bar_len - 1, pos))
            bar = [" "] * bar_len
            bar[bar_len // 2] = "|"
            bar[pos] = "█"
            bar_str = "".join(bar)

            sys.stdout.write(
                f"\r[Pitch: {pitch:+6.2f}°] [-25° {bar_str} +25°] | State: {label:<35} | Packets: {count}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down visualizer...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
