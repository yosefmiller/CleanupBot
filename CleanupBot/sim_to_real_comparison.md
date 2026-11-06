# CleanupBot: Sim-to-Real Telemetry Comparison Report
**Module 6 Demonstration Record** | *Timestamp: 2026-11-04 16:42:34*

## Summary Metrics
- **Total Rehearsals**: 4
- **Safety Auto-Rejections (Collision Gating)**: 0
- **Operator Approved & Executed**: 4
- **Operator Manual Rejections**: 0

## Sim-to-Real Telemetry Comparison Table

| Move / Command | Commanded $(v, \omega, t)$ | Gazebo Sim | Real Encoders | Real Tape & IMU | Status | Sim-to-Real Gap ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **turn_right (90.00 °)** | v=0.00m/s, w=-0.80r/s, t=2.0s | 0.000m, -91.0° | 0.006m | 0.000m, -101.6° | **[APPROVED] Executed** | $\Delta$Dist=0.000m, $\Delta$Turn=10.6° |
| **forward (0.30 m)** | v=0.20m/s, w=0.00r/s, t=1.5s | 0.326m, -0.0° | 0.310m | 0.310m, 8.4° | **[APPROVED] Executed** | $\Delta$Dist=0.016m, $\Delta$Turn=8.4° |
| **turn_left (90.00 °)** | v=0.00m/s, w=0.80r/s, t=2.0s | 0.000m, 91.4° | 0.005m | 0.000m, 106.0° | **[APPROVED] Executed** | $\Delta$Dist=0.000m, $\Delta$Turn=14.6° |
| **forward (0.40 m)** | v=0.15m/s, w=0.00r/s, t=2.7s | 0.415m, 0.0° | 0.435m | 0.435m, 22.4° | **[APPROVED] Executed** | $\Delta$Dist=0.020m, $\Delta$Turn=22.4° |

## Visual Accuracy Breakdown

### Move 1: turn_right (90.00 °)
```text
  Gazebo Turn:    [█████████████████████   ] -91.0°
  Real BNO085:    [████████████████████████] -101.6° (IMU Planar Yaw)
  Angular Delta:  10.6°
```

### Move 2: forward (0.30 m)
```text
  Gazebo Sim:     [████████████████████████] 0.326 m
  Real Encoders:  [██████████████████████  ] 0.310 m
  Real Tape Meas: [██████████████████████  ] 0.310 m
  Distance Delta: 0.016 m
  Heading Drift:  +8.4° (BNO085 Gyro Lock)
```

### Move 3: turn_left (90.00 °)
```text
  Gazebo Turn:    [████████████████████    ] +91.4°
  Real BNO085:    [████████████████████████] +106.0° (IMU Planar Yaw)
  Angular Delta:  14.6°
```

### Move 4: forward (0.40 m)
```text
  Gazebo Sim:     [██████████████████████  ] 0.415 m
  Real Encoders:  [████████████████████████] 0.435 m
  Real Tape Meas: [████████████████████████] 0.435 m
  Distance Delta: 0.020 m
  Heading Drift:  +22.4° (BNO085 Gyro Lock)
```
