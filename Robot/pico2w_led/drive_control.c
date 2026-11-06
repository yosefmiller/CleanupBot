/**
 * @file drive_control.c
 * @brief Differential drive kinematics, encoder closed-loop calibration, and watchdog safety.
 */

#include "drive_control.h"

#include <math.h>
#include "pico/stdlib.h"
#include "drv8833.h"
#include "encoder.h"

static absolute_time_t last_cmd_time = 0;
static bool is_active = false;
static int32_t current_status = 0;

// Commanded base speeds
static float target_v = 0.0f;
static float target_w = 0.0f;

// Active encoder drift compensation error and straight tracking baseline
static float dynamic_trim_correction = 0.0f;
static int32_t straight_start_l = 0;
static int32_t straight_start_r = 0;
static bool in_straight_mode = false;

static float speed_to_duty(float speed_mps, float trim_factor) {
    if (__builtin_fabsf(speed_mps) < 0.01f) {
        return 0.0f;
    }

    float sign = (speed_mps > 0.0f) ? 1.0f : -1.0f;
    float ratio = __builtin_fabsf(speed_mps) / MAX_WHEEL_SPEED_MPS;
    if (ratio > 1.0f) ratio = 1.0f;

    // Preserve MIN_DUTY as the absolute breakout floor so TT motor never stalls.
    // Scale only the dynamic headroom above MIN_DUTY by trim_factor.
    float dynamic_duty = (1.0f - MIN_DUTY) * ratio * trim_factor;
    float duty = MIN_DUTY + dynamic_duty;

    if (duty > 1.0f) duty = 1.0f;
    return sign * duty;
}

void drive_control_init(void) {
    drv8833_init();
    drv8833_stop();
    drv8833_gripper_standby();
    encoder_init();

    last_cmd_time = get_absolute_time();
    is_active = false;
    current_status = 0;
    target_v = 0.0f;
    target_w = 0.0f;
    dynamic_trim_correction = 0.0f;
    in_straight_mode = false;
    straight_start_l = 0;
    straight_start_r = 0;
}

void drive_control_stop(void) {
    drv8833_stop();
    is_active = false;
    current_status = 0;
    target_v = 0.0f;
    target_w = 0.0f;
    dynamic_trim_correction = 0.0f;
    in_straight_mode = false;
}

void drive_control_raw(float left_duty, float right_duty) {
    last_cmd_time = get_absolute_time();
    in_straight_mode = false;
    dynamic_trim_correction = 0.0f;
    target_v = 0.0f;
    target_w = 0.0f;

    if (__builtin_fabsf(left_duty) < 0.01f && __builtin_fabsf(right_duty) < 0.01f) {
        drive_control_stop();
        return;
    }

    drv8833_gripper_standby(); // Keep OWI-535 asleep while driving
    drv8833_set_wheel_speeds(left_duty, right_duty);
    is_active = true;
    current_status = 1;
}

void drive_control_cmd_vel(float linear_x, float angular_z) {
    if (linear_x > MAX_WHEEL_SPEED_MPS) linear_x = MAX_WHEEL_SPEED_MPS;
    if (linear_x < -MAX_WHEEL_SPEED_MPS) linear_x = -MAX_WHEEL_SPEED_MPS;
    if (angular_z > 1.50f) angular_z = 1.50f;
    if (angular_z < -1.50f) angular_z = -1.50f;

    last_cmd_time = get_absolute_time();
    target_v = linear_x;
    target_w = angular_z;

    if (__builtin_fabsf(linear_x) < 0.005f && __builtin_fabsf(angular_z) < 0.005f) {
        drive_control_stop();
        return;
    }

    // Check if entering straight drive mode
    if (__builtin_fabsf(angular_z) < 0.05f && __builtin_fabsf(linear_x) >= 0.02f) {
        if (!in_straight_mode) {
            encoder_get_ticks(&straight_start_l, &straight_start_r);
            in_straight_mode = true;
            dynamic_trim_correction = 0.0f;
        }
    } else {
        in_straight_mode = false;
        dynamic_trim_correction = 0.0f;
    }

    float half_track = WHEEL_SEPARATION_M / 2.0f;
    float v_left = linear_x - (angular_z * half_track);
    float v_right = linear_x + (angular_z * half_track);

    float left_duty = speed_to_duty(v_left, LEFT_TRIM_FACTOR - dynamic_trim_correction);
    float right_duty = speed_to_duty(v_right, RIGHT_TRIM_FACTOR + dynamic_trim_correction);

    drv8833_gripper_standby(); // Keep OWI-535 asleep while driving
    drv8833_set_wheel_speeds(left_duty, right_duty);

    is_active = true;
    current_status = 1;
}

void drive_control_update(void) {
    if (!is_active) {
        return;
    }

    // 1. Safety Watchdog Check (500 ms)
    int64_t elapsed_us = absolute_time_diff_us(last_cmd_time, get_absolute_time());
    if (elapsed_us > CMD_TIMEOUT_US) {
        drv8833_stop();
        is_active = false;
        current_status = -1;
        in_straight_mode = false;
        return;
    }

    // 2. Active Closed-Loop Wheel Calibration (active during straight moves)
    // Compares relative tick accumulation since the straight move began.
    if (in_straight_mode && __builtin_fabsf(target_v) >= 0.02f) {
        int32_t l_ticks = 0, r_ticks = 0;
        encoder_get_ticks(&l_ticks, &r_ticks);

        int32_t delta_l = l_ticks - straight_start_l;
        int32_t delta_r = r_ticks - straight_start_r;
        int32_t error = delta_l - delta_r; // Positive if left leads right

        // Proportional closed-loop correction (0.0003 per tick difference)
        float correction = (float)error * 0.0003f;
        if (correction > 0.15f) correction = 0.15f;
        if (correction < -0.15f) correction = -0.15f;
        dynamic_trim_correction = correction;

        // Reapply compensated duty
        float left_duty = speed_to_duty(target_v, LEFT_TRIM_FACTOR - dynamic_trim_correction);
        float right_duty = speed_to_duty(target_v, RIGHT_TRIM_FACTOR + dynamic_trim_correction);
        drv8833_set_wheel_speeds(left_duty, right_duty);
    }
}

int32_t drive_control_get_status(void) {
    return current_status;
}

float drive_control_get_distance_m(void) {
    return encoder_get_average_distance_m();
}

void drive_control_get_ticks(int32_t *left, int32_t *right) {
    encoder_get_ticks(left, right);
}
