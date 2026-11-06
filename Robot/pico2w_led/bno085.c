/**
 * @file bno085.c
 * @brief I2C Driver implementation for the Teyleten GY-BNO085 IMU using CEVA SHTP.
 */

#include "bno085.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hardware/gpio.h"
#include "hardware/i2c.h"
#include "pico/stdlib.h"

#define BNO085_I2C              i2c1
#define BNO085_SDA_PIN          14
#define BNO085_SCL_PIN          15
#define BNO085_I2C_BAUDRATE     100000

#define BNO085_ADDR_PRIMARY     0x4A
#define BNO085_ADDR_SECONDARY   0x4B
#define BNO085_TIMEOUT_US       35000

#define SHTP_CHAN_COMMAND       0
#define SHTP_CHAN_CONTROL       2
#define SHTP_CHAN_REPORTS       3

#define SH2_REPORT_BASE_TIMESTAMP       0xFB
#define SH2_REPORT_ROTATION_VECTOR      0x05
#define SH2_REPORT_GAME_ROTATION_VECTOR 0x08

#define SH2_CMD_SET_FEATURE             0xFD

#define Q14_SCALE               (1.0f / 16384.0f)
#define RAD_TO_DEG              (180.0f / 3.14159265358979323846f)

static uint8_t bno_address = BNO085_ADDR_SECONDARY;
static uint8_t shtp_sequence = 0;

static float last_qw = 1.0f;
static float last_qx = 0.0f;
static float last_qy = 0.0f;
static float last_qz = 0.0f;
static float last_heading_deg = 0.0f;
static float last_pitch_deg = 0.0f;
static absolute_time_t last_report_time = 0;
static bool has_valid_reading = false;

static int read_shtp_packet(uint8_t *buffer, size_t max_len) {
    if (max_len < 4) return -1;

    int bytes_read = i2c_read_timeout_us(
        BNO085_I2C, bno_address, buffer, max_len, false, BNO085_TIMEOUT_US
    );
    if (bytes_read < 4) return -1;

    uint16_t packet_len = (uint16_t)(buffer[0] | ((buffer[1] & 0x7F) << 8));
    if (packet_len < 4 || packet_len > max_len) return -1;

    return (int)packet_len;
}

static bool send_shtp_packet(uint8_t channel, const uint8_t *payload, uint16_t length) {
    uint8_t buffer[64];
    uint16_t total_length = length + 4;
    if (total_length > sizeof(buffer)) return false;

    buffer[0] = (uint8_t)(total_length & 0xFF);
    buffer[1] = (uint8_t)((total_length >> 8) & 0xFF);
    buffer[2] = channel;
    buffer[3] = shtp_sequence++;

    memcpy(&buffer[4], payload, length);

    int bytes_written = i2c_write_timeout_us(
        BNO085_I2C, bno_address, buffer, total_length, false, BNO085_TIMEOUT_US
    );
    return (bytes_written == (int)total_length);
}

static bool enable_feature_report(uint8_t report_id, uint32_t report_interval_us) {
    uint8_t cmd[21] = {0};
    cmd[0] = SH2_CMD_SET_FEATURE;
    cmd[1] = report_id;
    cmd[2] = 0;
    cmd[3] = 0;
    cmd[4] = 0;
    cmd[5] = (uint8_t)(report_interval_us & 0xFF);
    cmd[6] = (uint8_t)((report_interval_us >> 8) & 0xFF);
    cmd[7] = (uint8_t)((report_interval_us >> 16) & 0xFF);
    cmd[8] = (uint8_t)((report_interval_us >> 24) & 0xFF);

    return send_shtp_packet(SHTP_CHAN_CONTROL, cmd, sizeof(cmd));
}

static void parse_sensor_reports(const uint8_t *payload, uint16_t length) {
    uint16_t offset = 0;
    while (offset < length) {
        uint8_t report_id = payload[offset];

        if (report_id == SH2_REPORT_BASE_TIMESTAMP) {
            offset += 5;
            continue;
        }

        if (report_id == SH2_REPORT_GAME_ROTATION_VECTOR ||
            report_id == SH2_REPORT_ROTATION_VECTOR) {
            if (offset + 12 > length) break;

            int16_t raw_i = (int16_t)(payload[offset + 4] | (payload[offset + 5] << 8));
            int16_t raw_j = (int16_t)(payload[offset + 6] | (payload[offset + 7] << 8));
            int16_t raw_k = (int16_t)(payload[offset + 8] | (payload[offset + 9] << 8));
            int16_t raw_r = (int16_t)(payload[offset + 10] | (payload[offset + 11] << 8));

            float qx = (float)raw_i * Q14_SCALE;
            float qy = (float)raw_j * Q14_SCALE;
            float qz = (float)raw_k * Q14_SCALE;
            float qw = (float)raw_r * Q14_SCALE;

            last_qx = qx;
            last_qy = qy;
            last_qz = qz;
            last_qw = qw;

            // Planar Yaw Heading: rotation about vertical Z_imu axis
            float siny_cosp = 2.0f * (qw * qz + qx * qy);
            float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
            last_heading_deg = atan2f(siny_cosp, cosy_cosp) * RAD_TO_DEG;

            // Pitch: rotation about lateral X_imu axis (wheel axle)
            float sinp = 2.0f * (qw * qx + qy * qz);
            if (__builtin_fabsf(sinp) >= 1.0f) {
                last_pitch_deg = copysignf(90.0f, sinp);
            } else {
                last_pitch_deg = asinf(sinp) * RAD_TO_DEG;
            }

            last_report_time = get_absolute_time();
            has_valid_reading = true;

            offset += 14;
            continue;
        }

        break;
    }
}

bool bno085_init(void) {
    i2c_init(BNO085_I2C, BNO085_I2C_BAUDRATE);
    gpio_set_function(BNO085_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(BNO085_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(BNO085_SDA_PIN);
    gpio_pull_up(BNO085_SCL_PIN);

    uint8_t probe_byte = 0;
    bno_address = BNO085_ADDR_SECONDARY;
    if (i2c_read_timeout_us(BNO085_I2C, bno_address, &probe_byte, 1, false, 20000) < 0) {
        bno_address = BNO085_ADDR_PRIMARY;
        if (i2c_read_timeout_us(BNO085_I2C, bno_address, &probe_byte, 1, false, 20000) < 0) {
            printf("BNO085 not found at 0x4B or 0x4A.\n");
            return false;
        }
    }
    printf("BNO085 detected at 0x%02X\n", bno_address);

    sleep_ms(150);

    uint8_t rx_buffer[128];
    for (int i = 0; i < 5; i++) {
        read_shtp_packet(rx_buffer, sizeof(rx_buffer));
        sleep_ms(10);
    }

    if (!enable_feature_report(SH2_REPORT_GAME_ROTATION_VECTOR, 20000)) {
        printf("Failed to enable BNO085 Game Rotation Vector.\n");
        return false;
    }

    printf("BNO085 initialized: Game Rotation Vector stream active at 50 Hz.\n");
    return true;
}

bool bno085_read_rotation(
    float *qw,
    float *qx,
    float *qy,
    float *qz,
    float *heading_deg,
    float *pitch_deg
) {
    uint8_t rx_buffer[128];
    int len = read_shtp_packet(rx_buffer, sizeof(rx_buffer));
    if (len >= 4 && rx_buffer[2] == SHTP_CHAN_REPORTS) {
        parse_sensor_reports(&rx_buffer[4], (uint16_t)(len - 4));
    }

    if (!has_valid_reading) {
        return false;
    }

    if (absolute_time_diff_us(last_report_time, get_absolute_time()) > 500000) {
        return false;
    }

    if (qw) *qw = last_qw;
    if (qx) *qx = last_qx;
    if (qy) *qy = last_qy;
    if (qz) *qz = last_qz;
    if (heading_deg) *heading_deg = last_heading_deg;
    if (pitch_deg) *pitch_deg = last_pitch_deg;

    return true;
}

bool bno085_read_pitch(float *degrees) {
    return bno085_read_rotation(NULL, NULL, NULL, NULL, NULL, degrees);
}
