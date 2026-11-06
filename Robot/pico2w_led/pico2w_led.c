/**
 * @file pico2w_led.c
 * @brief Main firmware for Raspberry Pi Pico 2 W: micro-ROS, DRV8833 differential drive,
 *        BNO085 3D IMU telemetry, wheel encoder calibration, and OWI-535 safety isolation.
 *
 * Subscriptions:
 *   - /pico/cmd_vel        (geometry_msgs/msg/Twist): Linear and angular wheel velocity commands.
 *
 * Publishers:
 *   - /pico/imu/heading    (std_msgs/msg/Float32): Planar yaw heading in degrees [-180 .. +180].
 *   - /pico/imu/data       (sensor_msgs/msg/Imu): Full 3D orientation quaternion for eye-in-hand gripper.
 *   - /pico/drive/status   (std_msgs/msg/Int32): 0 = Idle, 1 = Moving, -1 = Watchdog Timeout.
 *   - /pico/drive/distance (std_msgs/msg/Float32): Live average wheel encoder distance in meters.
 *   - /pico/drive/encoders (geometry_msgs/msg/Vector3): Atomic left ticks (x), right ticks (y), distance (z).
 */

#include <stdio.h>
#include <string.h>

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "hardware/watchdog.h"

#include <rcl/error_handling.h>
#include <rcl/rcl.h>
#include <rclc/executor.h>
#include <rclc/rclc.h>
#include <rmw_microros/rmw_microros.h>

#include "geometry_msgs/msg/quaternion.h"
#include "geometry_msgs/msg/twist.h"
#include "geometry_msgs/msg/vector3.h"
#include "std_msgs/msg/float32.h"
#include "std_msgs/msg/int32.h"

#include "bno085.h"
#include "drive_control.h"
#include "drv8833.h"
#include "encoder.h"
#include "picow_udp_transport.h"
#include "wifi_config.h"

static rcl_publisher_t heading_publisher;
static rcl_publisher_t pitch_publisher;
static rcl_publisher_t orientation_publisher;
static rcl_publisher_t status_publisher;
static rcl_publisher_t distance_publisher;
static rcl_publisher_t encoders_publisher;
static rcl_subscription_t cmd_vel_subscription;

static geometry_msgs__msg__Twist cmd_vel_msg;
static std_msgs__msg__Float32 heading_msg;
static std_msgs__msg__Float32 pitch_msg;
static geometry_msgs__msg__Quaternion orientation_msg;
static std_msgs__msg__Int32 status_msg;
static std_msgs__msg__Float32 distance_msg;
static geometry_msgs__msg__Vector3 encoders_msg;

static picow_udp_transport_context_t transport_context;

static void cmd_vel_callback(const void *incoming) {
    const geometry_msgs__msg__Twist *twist = incoming;
    if ((float)twist->linear.y == 99.0f) {
        // Calibration Mode: direct raw PWM duties (x = left_duty, z = right_duty)
        drive_control_raw((float)twist->linear.x, (float)twist->angular.z);
    } else {
        drive_control_cmd_vel((float)twist->linear.x, (float)twist->angular.z);
    }
}

static bool check_rcl(rcl_ret_t result, const char *operation) {
    if (result == RCL_RET_OK) return true;
    printf("%s failed: %d\n", operation, (int)result);
    return false;
}

int main(void) {
    stdio_init_all();
    sleep_ms(2000);

    printf("\n=========================================\n");
    printf("  CleanupBot Pico 2 W micro-ROS Firmware\n");
    printf("=========================================\n");

    if (cyw43_arch_init()) {
        printf("Failed to initialize CYW43 Wi-Fi architecture.\n");
        return 1;
    }
    cyw43_arch_enable_sta_mode();

    // Initialize Motors, Encoders & BNO085
    drive_control_init();
    bool bno_ok = bno085_init();
    if (!bno_ok) {
        printf("WARNING: BNO085 IMU initialization failed or sensor not found.\n");
    }

    printf("Connecting to Wi-Fi SSID '%s'...\n", WIFI_SSID);
    if (cyw43_arch_wifi_connect_timeout_ms(
            WIFI_SSID,
            WIFI_PASSWORD,
            CYW43_AUTH_WPA2_AES_PSK,
            30000)) {
        printf("Wi-Fi connection failed. Check wifi_config.h settings.\n");
        return 1;
    }
    printf("Wi-Fi connected successfully!\n");

    memset(&transport_context, 0, sizeof(transport_context));
    transport_context.agent_ip = ROS_AGENT_IP;
    transport_context.agent_port = ROS_AGENT_PORT;

    rmw_ret_t transport_status = rmw_uros_set_custom_transport(
        false,
        &transport_context,
        picow_udp_transport_open,
        picow_udp_transport_close,
        picow_udp_transport_write,
        picow_udp_transport_read
    );
    if (transport_status != RMW_RET_OK) {
        printf("Failed to set micro-ROS custom UDP transport.\n");
        return 1;
    }

    printf("Pinging micro-ROS agent at %s:%d...\n", ROS_AGENT_IP, ROS_AGENT_PORT);
    while (rmw_uros_ping_agent(1000, 1) != RMW_RET_OK) {
        printf("Waiting for micro-ROS agent...\n");
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(250);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(250);
    }
    printf("micro-ROS agent responded! Initializing ROS 2 node...\n");

    rcl_allocator_t allocator = rcl_get_default_allocator();
    rclc_support_t support;
    rcl_node_t node = rcl_get_zero_initialized_node();

    if (!check_rcl(rclc_support_init(&support, 0, NULL, &allocator), "rclc_support_init") ||
        !check_rcl(rclc_node_init_default(&node, "cleanupbot_pico", "", &support), "rclc_node_init")) {
        return 1;
    }

    // Initialize Telemetry Publishers (Best Effort QoS for non-blocking real-time streaming)
    heading_publisher = rcl_get_zero_initialized_publisher();
    pitch_publisher = rcl_get_zero_initialized_publisher();
    orientation_publisher = rcl_get_zero_initialized_publisher();
    status_publisher = rcl_get_zero_initialized_publisher();
    distance_publisher = rcl_get_zero_initialized_publisher();
    encoders_publisher = rcl_get_zero_initialized_publisher();

    check_rcl(
        rclc_publisher_init_best_effort(
            &heading_publisher,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
            "/pico/imu/heading"
        ),
        "heading_publisher init"
    );

    check_rcl(
        rclc_publisher_init_best_effort(
            &pitch_publisher,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
            "/pico/imu/pitch"
        ),
        "pitch_publisher init"
    );

    check_rcl(
        rclc_publisher_init_best_effort(
            &orientation_publisher,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Quaternion),
            "/pico/imu/orientation"
        ),
        "orientation_publisher init"
    );

    check_rcl(
        rclc_publisher_init_best_effort(
            &status_publisher,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
            "/pico/drive/status"
        ),
        "status_publisher init"
    );

    check_rcl(
        rclc_publisher_init_best_effort(
            &distance_publisher,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
            "/pico/drive/distance"
        ),
        "distance_publisher init"
    );

    check_rcl(
        rclc_publisher_init_best_effort(
            &encoders_publisher,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Vector3),
            "/pico/drive/encoders"
        ),
        "encoders_publisher init"
    );

    // Initialize Subscription
    cmd_vel_subscription = rcl_get_zero_initialized_subscription();
    check_rcl(
        rclc_subscription_init_best_effort(
            &cmd_vel_subscription,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
            "/pico/cmd_vel"
        ),
        "cmd_vel_subscription init"
    );

    // Executor (1 subscription)
    rclc_executor_t executor = rclc_executor_get_zero_initialized_executor();
    check_rcl(rclc_executor_init(&executor, &support.context, 1, &allocator), "executor init");
    check_rcl(
        rclc_executor_add_subscription(
            &executor,
            &cmd_vel_subscription,
            &cmd_vel_msg,
            cmd_vel_callback,
            ON_NEW_DATA
        ),
        "add cmd_vel subscription"
    );

    printf("micro-ROS node ready. Listening on /pico/cmd_vel.\n");
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);

    // Stagger telemetry to prevent CYW43 Wi-Fi SPI TX FIFO queue drops
    absolute_time_t next_telemetry_nav = make_timeout_time_ms(50);   // 20 Hz (Nav: Encoders + Yaw)
    absolute_time_t next_telemetry_aux = make_timeout_time_ms(75);   // 10 Hz (Aux: Pitch, Quat, Status)
    absolute_time_t next_serial_log = make_timeout_time_ms(2000);    // 0.5 Hz Diagnostic Log

    static float cached_heading = 0.0f;
    static float cached_pitch = 0.0f;
    static float cached_qw = 1.0f, cached_qx = 0.0f, cached_qy = 0.0f, cached_qz = 0.0f;

    while (true) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
        drive_control_update();

        // -------------------------------------------------------------
        // Phase 1 (20 Hz): High-Priority Navigation Telemetry
        // Encoders and Yaw heading are published FIRST to eliminate starvation
        // -------------------------------------------------------------
        if (time_reached(next_telemetry_nav)) {
            // Read fresh BNO085 orientation
            float qw, qx, qy, qz, heading, pitch;
            if (bno085_read_rotation(&qw, &qx, &qy, &qz, &heading, &pitch)) {
                cached_heading = heading;
                cached_pitch = pitch;
                cached_qw = qw;
                cached_qx = qx;
                cached_qy = qy;
                cached_qz = qz;
            }

            // 1. Encoders (Left Ticks, Right Ticks, Distance) - Priority #1
            int32_t lt = 0, rt = 0;
            drive_control_get_ticks(&lt, &rt);
            float dist_m = drive_control_get_distance_m();
            encoders_msg.x = (double)lt;
            encoders_msg.y = (double)rt;
            encoders_msg.z = (double)dist_m;
            (void)rcl_publish(&encoders_publisher, &encoders_msg, NULL);

            // 2. Planar Yaw Heading - Priority #2
            heading_msg.data = cached_heading;
            (void)rcl_publish(&heading_publisher, &heading_msg, NULL);

            next_telemetry_nav = make_timeout_time_ms(50);
        }

        // -------------------------------------------------------------
        // Phase 2 (10 Hz, 25ms interleaved): Auxiliary Telemetry
        // -------------------------------------------------------------
        if (time_reached(next_telemetry_aux)) {
            // 3. Gripper / Arm Pitch Angle
            pitch_msg.data = cached_pitch;
            (void)rcl_publish(&pitch_publisher, &pitch_msg, NULL);

            // 4. 3D Orientation Quaternion for Odometry / EKF Sensor Fusion
            orientation_msg.w = (double)cached_qw;
            orientation_msg.x = (double)cached_qx;
            orientation_msg.y = (double)cached_qy;
            orientation_msg.z = (double)cached_qz;
            (void)rcl_publish(&orientation_publisher, &orientation_msg, NULL);

            // 5. Drive Status & Cumulative Distance
            status_msg.data = drive_control_get_status();
            (void)rcl_publish(&status_publisher, &status_msg, NULL);

            distance_msg.data = drive_control_get_distance_m();
            (void)rcl_publish(&distance_publisher, &distance_msg, NULL);

            next_telemetry_aux = make_timeout_time_ms(100);
        }

        // Diagnostic serial print every 2.0s
        if (time_reached(next_serial_log)) {
            int32_t lt = 0, rt = 0;
            drive_control_get_ticks(&lt, &rt);
            printf("[TELEMETRY STREAM] Yaw: %+.1f° | Pitch: %+.1f° | Ticks: L=%ld, R=%ld\n",
                   cached_heading, cached_pitch, (long)lt, (long)rt);
            next_serial_log = make_timeout_time_ms(2000);
        }
    }

    return 0;
}
