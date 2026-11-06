/**
 * @file encoder.c
 * @brief Quadrature wheel encoder driver implementation using RP2350 GPIO interrupts.
 *
 * GUIDE FOR DEVELOPERS:
 * 1. How Quadrature Encoding Works:
 *    - Each wheel has two sensor channels (Phase A and Phase B) positioned 90° out of phase.
 *    - As the wheel rotates:
 *        * Rotating FORWARD : Phase A changes state BEFORE Phase B.
 *        * Rotating REVERSE : Phase B changes state BEFORE Phase A.
 *    - By checking the state of Phase B when Phase A transitions, we know the exact direction!
 *
 * 2. Hardware Interrupts (`gpio_set_irq_enabled_with_callback`):
 *    - Instead of wasting CPU cycles constantly polling pins in a loop, the RP2350 hardware
 *      triggers an Interrupt Service Routine (ISR) the instant a voltage edge occurs.
 *    - This guarantees zero missed counts even at full wheel speed.
 *
 * 3. Volatile Variables:
 *    - Variables modified inside an ISR and read by the main loop must be marked `volatile`.
 *    - This tells the C compiler: "Do NOT cache this variable in a CPU register because an
 *      interrupt can change its value in RAM at any microsecond."
 */

#include "encoder.h"

#include <math.h>
#include "hardware/gpio.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"

// ----------------------------------------------------------------------------
// Pinout Definitions
// ----------------------------------------------------------------------------
#define ENCODER_LEFT_A_PIN   18   // GP18 - Left motor encoder Phase A
#define ENCODER_LEFT_B_PIN   19   // GP19 - Left motor encoder Phase B
#define ENCODER_RIGHT_A_PIN  20   // GP20 - Right motor encoder Phase A
#define ENCODER_RIGHT_B_PIN  21   // GP21 - Right motor encoder Phase B

// Meters per tick constant
// Circumference = 2 * PI * R
// Meters/tick = Circumference / TICKS_PER_REV
static const float CIRCUMFERENCE_M = 2.0f * 3.14159265358979323846f * ENCODER_WHEEL_RADIUS_M;

// ----------------------------------------------------------------------------
// Volatile Tick Counters (Updated asynchronously by GPIO ISR)
// ----------------------------------------------------------------------------
static volatile int32_t left_ticks = 0;
static volatile int32_t right_ticks = 0;

// Encoder Direction Configuration:
// Set to true if forward physical wheel rolling produces negative tick counts.
#define INVERT_LEFT_ENCODER   true
#define INVERT_RIGHT_ENCODER  true

/**
 * @brief Shared GPIO Interrupt Service Routine (ISR) for all encoder pins.
 *
 * Called automatically by hardware whenever an edge transition occurs on GP18-GP21.
 */
static void encoder_gpio_isr(uint gpio, uint32_t events) {
    // --- Left Wheel Encoder (GP18 Phase A) ---
    if (gpio == ENCODER_LEFT_A_PIN) {
        bool a_val = gpio_get(ENCODER_LEFT_A_PIN);
        bool b_val = gpio_get(ENCODER_LEFT_B_PIN);
#if INVERT_LEFT_ENCODER
        if (a_val != b_val) {
            left_ticks++;
        } else {
            left_ticks--;
        }
#else
        if (a_val == b_val) {
            left_ticks++;
        } else {
            left_ticks--;
        }
#endif
    }

    // --- Right Wheel Encoder (GP20 Phase A) ---
    else if (gpio == ENCODER_RIGHT_A_PIN) {
        bool a_val = gpio_get(ENCODER_RIGHT_A_PIN);
        bool b_val = gpio_get(ENCODER_RIGHT_B_PIN);
#if INVERT_RIGHT_ENCODER
        if (a_val == b_val) {
            right_ticks++;
        } else {
            right_ticks--;
        }
#else
        if (a_val != b_val) {
            right_ticks++;
        } else {
            right_ticks--;
        }
#endif
    }
}

void encoder_init(void) {
    const uint pins[] = {
        ENCODER_LEFT_A_PIN,
        ENCODER_LEFT_B_PIN,
        ENCODER_RIGHT_A_PIN,
        ENCODER_RIGHT_B_PIN,
    };

    // Configure pins as inputs with internal pull-up resistors
    for (size_t i = 0; i < sizeof(pins) / sizeof(pins[0]); i++) {
        gpio_init(pins[i]);
        gpio_set_dir(pins[i], GPIO_IN);
        gpio_pull_up(pins[i]);
    }

    // Attach hardware edge interrupts to Phase A of both wheels
    // We trigger on both RISING and FALLING edges to double measurement resolution
    gpio_set_irq_enabled_with_callback(
        ENCODER_LEFT_A_PIN,
        GPIO_IRQ_EDGE_RISE | GPIO_IRQ_EDGE_FALL,
        true,
        &encoder_gpio_isr
    );

    gpio_set_irq_enabled(
        ENCODER_RIGHT_A_PIN,
        GPIO_IRQ_EDGE_RISE | GPIO_IRQ_EDGE_FALL,
        true
    );

    left_ticks = 0;
    right_ticks = 0;
}

void encoder_reset(void) {
    uint32_t irq_status = save_and_disable_interrupts();
    left_ticks = 0;
    right_ticks = 0;
    restore_interrupts(irq_status);
}

void encoder_get_ticks(int32_t *out_left, int32_t *out_right) {
    // Atomic read: disable interrupts momentarily to prevent a tick occurring mid-read
    uint32_t irq_status = save_and_disable_interrupts();
    int32_t l = left_ticks;
    int32_t r = right_ticks;
    restore_interrupts(irq_status);

    if (out_left) *out_left = l;
    if (out_right) *out_right = r;
}

void encoder_get_distances_m(float *left_dist_m, float *right_dist_m) {
    int32_t l, r;
    encoder_get_ticks(&l, &r);

    float meters_per_tick = CIRCUMFERENCE_M / ENCODER_TICKS_PER_REV;

    if (left_dist_m) *left_dist_m = (float)l * meters_per_tick;
    if (right_dist_m) *right_dist_m = (float)r * meters_per_tick;
}

float encoder_get_average_distance_m(void) {
    float l, r;
    encoder_get_distances_m(&l, &r);
    return (l + r) / 2.0f;
}
