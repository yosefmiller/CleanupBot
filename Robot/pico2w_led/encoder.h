/**
 * @file encoder.h
 * @brief Quadrature wheel encoder driver for Raspberry Pi Pico 2 W (RP2350).
 *
 * Hardware Pinout:
 *   - GP18 : Left Wheel Encoder Phase A
 *   - GP19 : Left Wheel Encoder Phase B
 *   - GP20 : Right Wheel Encoder Phase A
 *   - GP21 : Right Wheel Encoder Phase B
 *
 * Uses hardware GPIO edge interrupts for zero-CPU-overhead background counting.
 * Provides atomic tick reads and distance conversion for closed-loop calibration.
 */

#ifndef ENCODER_H
#define ENCODER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Encoder resolution configuration
// CleanupBot TT Motor: 12 PPR Hall encoder * 2 edges (rise+fall) * 1:90 gearbox = 2160 ticks/wheel rev
#ifndef ENCODER_TICKS_PER_REV
#define ENCODER_TICKS_PER_REV   2160.0f
#endif

#define ENCODER_WHEEL_RADIUS_M  0.033f   // 3.3 cm wheel radius (6.6 cm diameter)

/**
 * @brief Initialize encoder GPIO pins with pull-ups and configure edge interrupts.
 */
void encoder_init(void);

/**
 * @brief Reset both left and right tick counters to zero.
 */
void encoder_reset(void);

/**
 * @brief Atomically read the raw signed tick counts for both wheels.
 *
 * @param[out] left_ticks  Pointer to int32_t storing left wheel tick count.
 * @param[out] right_ticks Pointer to int32_t storing right wheel tick count.
 */
void encoder_get_ticks(int32_t *left_ticks, int32_t *right_ticks);

/**
 * @brief Calculate the linear distance traveled by each wheel in meters.
 *
 * @param[out] left_dist_m  Pointer to float storing left wheel distance.
 * @param[out] right_dist_m Pointer to float storing right wheel distance.
 */
void encoder_get_distances_m(float *left_dist_m, float *right_dist_m);

/**
 * @brief Get the average forward linear distance traveled by the robot base in meters.
 *
 * @return Forward translation (left + right) / 2.0 in meters.
 */
float encoder_get_average_distance_m(void);

#ifdef __cplusplus
}
#endif

#endif // ENCODER_H
