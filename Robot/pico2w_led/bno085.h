/**
 * @file bno085.h
 * @brief Driver interface for Teyleten Robot GY-BNO085 9-DOF IMU module.
 *
 * Hardware Interface:
 *   - I2C Peripheral : I2C1
 *   - SDA Pin        : GP14
 *   - SCL Pin        : GP15
 *   - Clock Speed    : 100 kHz (I2C Standard-mode)
 *
 * Features:
 *   - Game Rotation Vector (Report 0x08): Fuses 3-axis gyro + 3-axis accelerometer.
 *     Immune to indoor magnetic anomalies, providing rock-solid drift-free yaw.
 *   - Rotation Vector (Report 0x05): Absolute 3D orientation quaternion.
 */

#ifndef BNO085_H
#define BNO085_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the BNO085 on I2C1 (GP14 SDA, GP15 SCL).
 *
 * @return true if initialization succeeded; false on timeout or I2C bus error.
 */
bool bno085_init(void);

/**
 * @brief Read the current robot pitch angle in degrees (backward compatibility).
 */
bool bno085_read_pitch(float *degrees);

/**
 * @brief Read the latest 3D orientation quaternion and calculated Euler angles.
 *
 * @param[out] qw          Pointer to float storing quaternion W component.
 * @param[out] qx          Pointer to float storing quaternion X component.
 * @param[out] qy          Pointer to float storing quaternion Y component.
 * @param[out] qz          Pointer to float storing quaternion Z component.
 * @param[out] heading_deg Pointer to float storing planar yaw heading in degrees [-180.0 .. +180.0].
 * @param[out] pitch_deg   Pointer to float storing arm pitch in degrees.
 *
 * @return true if fresh orientation telemetry was decoded; false otherwise.
 */
bool bno085_read_rotation(
    float *qw,
    float *qx,
    float *qy,
    float *qz,
    float *heading_deg,
    float *pitch_deg
);

#ifdef __cplusplus
}
#endif

#endif // BNO085_H
