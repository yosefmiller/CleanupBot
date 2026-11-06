/**
 * @file drv8833.h
 * @brief Hardware driver interface for dual DRV8833 H-Bridge DC Motor Driver Modules.
 *
 * Hardware Wiring:
 *   Module 1 (Chassis Drive Wheels):
 *     - GP9  : STBY (Wheels sleep/wake)        [white]
 *     - GP11 : AIN1 (Left wheel motor PWM)      [yellow]
 *     - GP10 : AIN2 (Left wheel motor PWM)      [brown]
 *     - GP12 : BIN1 (Right wheel motor PWM)     [yellow]
 *     - GP13 : BIN2 (Right wheel motor PWM)     [brown]
 *
 *   Module 2 (OWI-535 Gripper Actuators):
 *     - GP8  : STBY (Gripper sleep/wake)        [white]
 *     - GP3  : AIN1 (Gripper jaw motor PWM)     [yellow]
 *     - GP2  : AIN2 (Gripper jaw motor PWM)     [orange]
 *     - GP4  : BIN1 (Gripper tilt motor PWM)    [yellow]
 *     - GP5  : BIN2 (Gripper tilt motor PWM)    [orange]
 *
 * CRITICAL OWI-535 ELECTRICAL SAFETY:
 *   - The OWI-535 robotic arm uses 3.0V rated toy DC motors without limit switches.
 *   - Powered from a 6xAA battery pack (7.2V - 9.0V), applying 100% duty cycle would
 *     burn out the motor windings, strip plastic gears, and draw 1.2A stall currents.
 *   - All gripper pins use hardware PWM with MAX_GRIPPER_DUTY clamped to 0.35 (35%).
 *     7.5V * 0.35 = 2.62V nominal safe operating voltage!
 *   - Pulses are hardware-clamped to a 400ms maximum duration with immediate standby sleep.
 */

#ifndef DRV8833_H
#define DRV8833_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// OWI-535 Safety Limits
#define MAX_GRIPPER_DUTY        0.35f   // 35% PWM maximum: clamps 8V battery to safe 2.8V motor voltage
#define MAX_GRIPPER_PULSE_MS    400     // 400 ms maximum pulse duration: prevents mechanical jam & gear stripping

/**
 * @brief Initialize all 10 motor control GPIO pins and configure hardware PWM for wheels & gripper.
 */
void drv8833_init(void);

/**
 * @brief Stop wheel motors immediately and place the wheel DRV8833 into standby (STBY=LOW).
 */
void drv8833_stop(void);

/**
 * @brief Set independent signed duty cycles on left and right chassis drive wheels.
 *
 * @param left_duty  Signed duty cycle for left wheel  [-1.0f .. 1.0f]
 * @param right_duty Signed duty cycle for right wheel [-1.0f .. 1.0f]
 */
void drv8833_set_wheel_speeds(float left_duty, float right_duty);

/**
 * @brief Put the OWI-535 gripper DRV8833 into low-power sleep (STBY=LOW).
 */
void drv8833_gripper_standby(void);

/**
 * @brief Actuate OWI-535 jaw motor with strict voltage and duration clamping.
 *
 * @param close        true = close jaws; false = open jaws.
 * @param duty         Requested duty cycle (automatically clamped to MAX_GRIPPER_DUTY).
 * @param duration_ms  Pulse duration in ms (automatically clamped to MAX_GRIPPER_PULSE_MS).
 */
void drv8833_pulse_jaw(bool close, float duty, uint32_t duration_ms);

/**
 * @brief Actuate OWI-535 wrist tilt motor with strict voltage and duration clamping.
 *
 * @param down         true = tilt downward; false = tilt upward.
 * @param duty         Requested duty cycle (automatically clamped to MAX_GRIPPER_DUTY).
 * @param duration_ms  Pulse duration in ms (automatically clamped to MAX_GRIPPER_PULSE_MS).
 */
void drv8833_pulse_tilt(bool down, float duty, uint32_t duration_ms);

#ifdef __cplusplus
}
#endif

#endif // DRV8833_H
