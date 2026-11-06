/**
 * @file drive_control.h
 * @brief Differential drive kinematics, encoder closed-loop calibration, and watchdog safety.
 *
 * Implements:
 *   - 2-wheel differential drive unicycle kinematics
 *   - Deadband offset (MIN_DUTY) to overcome TT motor static friction
 *   - Real-time closed-loop encoder drift correction (keeps vehicle driving straight)
 *   - 500 ms communication watchdog (auto-stops wheels if ROS 2 commands stall)
 *   - OWI-535 gripper standby isolation
 */

#ifndef DRIVE_CONTROL_H
#define DRIVE_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CleanupBot Physical Kinematics (from CAD & config.py)
#define WHEEL_SEPARATION_M      0.140f   // 14.0 cm center-to-center track width
#define WHEEL_RADIUS_M          0.033f   // 3.3 cm wheel radius (6.6 cm diameter)

// Motor Calibration Parameters (Empirically measured via automated sweep)
#define MAX_WHEEL_SPEED_MPS     0.30f    // Speed ceiling in meters/second
#define MIN_DUTY                0.49f    // Overcomes 46% forward stiction of half-metal gearbox
#define LEFT_TRIM_FACTOR        0.97f    // Balanced Left speed scale (3.2% measured imbalance)
#define RIGHT_TRIM_FACTOR       1.00f    // Balanced Right speed scale

// Safety Watchdog
#define CMD_TIMEOUT_US          500000   // 500 ms timeout before automatic failsafe stop

/**
 * @brief Initialize the differential drive subsystem, encoders, and clear watchdog.
 */
void drive_control_init(void);

/**
 * @brief Process an incoming geometry_msgs/Twist command.
 *
 * @param linear_x  Forward/reverse linear speed in m/s (clamped to [-0.30, +0.30]).
 * @param angular_z Left/right angular speed in rad/s (clamped to [-1.50, +1.50]).
 */
void drive_control_cmd_vel(float linear_x, float angular_z);

/**
 * @brief Periodic update function (called in main loop every 10-20 ms).
 *
 * Enforces the 500 ms communication watchdog and applies active encoder drift correction.
 */
void drive_control_update(void);

/**
 * @brief Immediately halt both drive wheels.
 */
void drive_control_stop(void);

/**
 * @brief Directly command raw PWM duty cycles to the wheels (for calibration sweeps).
 *
 * @param left_duty  Signed duty cycle for left wheel [-1.0 .. +1.0].
 * @param right_duty Signed duty cycle for right wheel [-1.0 .. +1.0].
 */
void drive_control_raw(float left_duty, float right_duty);

/**
 * @brief Get the current drive status code.
 *
 * @return 0 = Idle / Stopped; 1 = Driving / Moving; -1 = Watchdog Timeout.
 */
int32_t drive_control_get_status(void);

/**
 * @brief Get the total linear distance traveled in meters (measured via wheel encoders).
 */
float drive_control_get_distance_m(void);

/**
 * @brief Get the current encoder tick counts.
 */
void drive_control_get_ticks(int32_t *left, int32_t *right);

#ifdef __cplusplus
}
#endif

#endif // DRIVE_CONTROL_H
