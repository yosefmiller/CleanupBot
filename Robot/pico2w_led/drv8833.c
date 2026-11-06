/**
 * @file drv8833.c
 * @brief Hardware driver implementation for dual DRV8833 H-Bridge DC Motor Driver Modules.
 *
 * Implements hardware PWM on all 8 motor drive pins to prevent overvoltage on the 3V OWI-535 motors,
 * and maintains low-power standby isolation on both DRV8833 chips.
 */

#include "drv8833.h"

#include <stddef.h>
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"

// Drive Wheels (DRV8833 #1)
#define WHEEL_STBY_PIN       9   // GP9  - STBY (wheels)
#define WHEEL_AIN1_PIN      11   // GP11 - AIN1 (left wheel)
#define WHEEL_AIN2_PIN      10   // GP10 - AIN2 (left wheel)
#define WHEEL_BIN1_PIN      12   // GP12 - BIN1 (right wheel)
#define WHEEL_BIN2_PIN      13   // GP13 - BIN2 (right wheel)

// Gripper Actuators (DRV8833 #2)
#define GRIPPER_STBY_PIN     8   // GP8  - STBY (gripper)
#define GRIPPER_JAW_AIN1     3   // GP3  - AIN1 (gripper jaw)
#define GRIPPER_JAW_AIN2     2   // GP2  - AIN2 (gripper jaw)
#define GRIPPER_TILT_BIN1    4   // GP4  - BIN1 (gripper tilt)
#define GRIPPER_TILT_BIN2    5   // GP5  - BIN2 (gripper tilt)

// Direction Inversion Flags:
// Set to true if a forward velocity command causes backward physical wheel rotation.
#define INVERT_LEFT_WHEEL    false
#define INVERT_RIGHT_WHEEL   false

// PWM Frequency: 125 MHz sysclk / 125.0 div / 1000 wrap = 1.0 kHz
#define PWM_WRAP_VAL      1000

/**
 * @brief Helper to initialize a GPIO pin as hardware PWM with 1 kHz carrier.
 */
static void init_pwm_pin(uint pin) {
    gpio_set_function(pin, GPIO_FUNC_PWM);
    uint slice = pwm_gpio_to_slice_num(pin);
    pwm_set_wrap(slice, PWM_WRAP_VAL);
    pwm_set_clkdiv(slice, 125.0f);
    pwm_set_enabled(slice, true);
    pwm_set_gpio_level(pin, 0);
}

void drv8833_init(void) {
    // 1. Initialize Standby sleep pins as digital outputs
    gpio_init(WHEEL_STBY_PIN);
    gpio_set_dir(WHEEL_STBY_PIN, GPIO_OUT);
    gpio_put(WHEEL_STBY_PIN, false); // Sleep

    gpio_init(GRIPPER_STBY_PIN);
    gpio_set_dir(GRIPPER_STBY_PIN, GPIO_OUT);
    gpio_put(GRIPPER_STBY_PIN, false); // Sleep

    // 2. Configure 4 wheel drive pins for hardware PWM
    init_pwm_pin(WHEEL_AIN1_PIN);
    init_pwm_pin(WHEEL_AIN2_PIN);
    init_pwm_pin(WHEEL_BIN1_PIN);
    init_pwm_pin(WHEEL_BIN2_PIN);

    // 3. Configure 4 gripper pins for hardware PWM (enables voltage scaling for 3V motors)
    init_pwm_pin(GRIPPER_JAW_AIN1);
    init_pwm_pin(GRIPPER_JAW_AIN2);
    init_pwm_pin(GRIPPER_TILT_BIN1);
    init_pwm_pin(GRIPPER_TILT_BIN2);
}

void drv8833_stop(void) {
    pwm_set_gpio_level(WHEEL_AIN1_PIN, 0);
    pwm_set_gpio_level(WHEEL_AIN2_PIN, 0);
    pwm_set_gpio_level(WHEEL_BIN1_PIN, 0);
    pwm_set_gpio_level(WHEEL_BIN2_PIN, 0);
    gpio_put(WHEEL_STBY_PIN, false); // Put wheel DRV8833 into sleep
}

void drv8833_gripper_standby(void) {
    pwm_set_gpio_level(GRIPPER_JAW_AIN1, 0);
    pwm_set_gpio_level(GRIPPER_JAW_AIN2, 0);
    pwm_set_gpio_level(GRIPPER_TILT_BIN1, 0);
    pwm_set_gpio_level(GRIPPER_TILT_BIN2, 0);
    gpio_put(GRIPPER_STBY_PIN, false); // Put gripper DRV8833 into sleep
}

void drv8833_set_wheel_speeds(float left_duty, float right_duty) {
    if (left_duty > -0.01f && left_duty < 0.01f &&
        right_duty > -0.01f && right_duty < 0.01f) {
        drv8833_stop();
        return;
    }

    if (left_duty > 1.0f) left_duty = 1.0f;
    if (left_duty < -1.0f) left_duty = -1.0f;
    if (right_duty > 1.0f) right_duty = 1.0f;
    if (right_duty < -1.0f) right_duty = -1.0f;

#if INVERT_LEFT_WHEEL
    left_duty = -left_duty;
#endif

#if INVERT_RIGHT_WHEEL
    right_duty = -right_duty;
#endif

    // Wake wheel driver
    gpio_put(WHEEL_STBY_PIN, true);
    sleep_us(50);

    // Right Wheel (BIN1, BIN2)
    uint16_t r_level = (uint16_t)(__builtin_fabsf(right_duty) * (float)PWM_WRAP_VAL);
    if (right_duty > 0.01f) {
        pwm_set_gpio_level(WHEEL_BIN1_PIN, r_level);
        pwm_set_gpio_level(WHEEL_BIN2_PIN, 0);
    } else if (right_duty < -0.01f) {
        pwm_set_gpio_level(WHEEL_BIN1_PIN, 0);
        pwm_set_gpio_level(WHEEL_BIN2_PIN, r_level);
    } else {
        pwm_set_gpio_level(WHEEL_BIN1_PIN, 0);
        pwm_set_gpio_level(WHEEL_BIN2_PIN, 0);
    }

    // Left Wheel (AIN1, AIN2) - mirrored
    uint16_t l_level = (uint16_t)(__builtin_fabsf(left_duty) * (float)PWM_WRAP_VAL);
    if (left_duty > 0.01f) {
        pwm_set_gpio_level(WHEEL_AIN1_PIN, 0);
        pwm_set_gpio_level(WHEEL_AIN2_PIN, l_level);
    } else if (left_duty < -0.01f) {
        pwm_set_gpio_level(WHEEL_AIN1_PIN, l_level);
        pwm_set_gpio_level(WHEEL_AIN2_PIN, 0);
    } else {
        pwm_set_gpio_level(WHEEL_AIN1_PIN, 0);
        pwm_set_gpio_level(WHEEL_AIN2_PIN, 0);
    }
}

void drv8833_pulse_jaw(bool close, float duty, uint32_t duration_ms) {
    if (duty < 0.05f) return;
    // Strict safety clamp: never exceed MAX_GRIPPER_DUTY (35%) to protect 3V motor
    if (duty > MAX_GRIPPER_DUTY) duty = MAX_GRIPPER_DUTY;
    // Strict duration clamp: never exceed MAX_GRIPPER_PULSE_MS (400ms) to prevent jamming
    if (duration_ms > MAX_GRIPPER_PULSE_MS) duration_ms = MAX_GRIPPER_PULSE_MS;

    uint16_t level = (uint16_t)(duty * (float)PWM_WRAP_VAL);

    gpio_put(GRIPPER_STBY_PIN, true);
    sleep_us(50);

    if (close) {
        pwm_set_gpio_level(GRIPPER_JAW_AIN1, level);
        pwm_set_gpio_level(GRIPPER_JAW_AIN2, 0);
    } else {
        pwm_set_gpio_level(GRIPPER_JAW_AIN1, 0);
        pwm_set_gpio_level(GRIPPER_JAW_AIN2, level);
    }

    sleep_ms(duration_ms);

    // Immediate de-energize and standby sleep
    drv8833_gripper_standby();
}

void drv8833_pulse_tilt(bool down, float duty, uint32_t duration_ms) {
    if (duty < 0.05f) return;
    // Strict safety clamp: never exceed MAX_GRIPPER_DUTY (35%) to protect 3V motor
    if (duty > MAX_GRIPPER_DUTY) duty = MAX_GRIPPER_DUTY;
    // Strict duration clamp: never exceed MAX_GRIPPER_PULSE_MS (400ms) to prevent jamming
    if (duration_ms > MAX_GRIPPER_PULSE_MS) duration_ms = MAX_GRIPPER_PULSE_MS;

    uint16_t level = (uint16_t)(duty * (float)PWM_WRAP_VAL);

    gpio_put(GRIPPER_STBY_PIN, true);
    sleep_us(50);

    if (down) {
        pwm_set_gpio_level(GRIPPER_TILT_BIN1, level);
        pwm_set_gpio_level(GRIPPER_TILT_BIN2, 0);
    } else {
        pwm_set_gpio_level(GRIPPER_TILT_BIN1, 0);
        pwm_set_gpio_level(GRIPPER_TILT_BIN2, level);
    }

    sleep_ms(duration_ms);

    // Immediate de-energize and standby sleep
    drv8833_gripper_standby();
}
