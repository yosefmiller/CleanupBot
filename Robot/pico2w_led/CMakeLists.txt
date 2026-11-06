cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Pico VS Code extension configuration
if(WIN32)
  set(USERHOME $ENV{USERPROFILE})
else()
  set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
  include(${picoVscode})
endif()

set(PICO_BOARD pico2_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)

project(pico2w_led C CXX ASM)

pico_sdk_init()

add_executable(pico2w_led
  bno085.c
  drive_control.c
  drv8833.c
  encoder.c
  microros_time.c
  pico2w_led.c
  picow_udp_transport.c
)

# Link Cortex-M33 micro-ROS static library archive
add_library(micro_ros STATIC IMPORTED)
set_target_properties(micro_ros PROPERTIES
  IMPORTED_LOCATION
    "${CMAKE_CURRENT_LIST_DIR}/libmicroros/libmicroros.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_LIST_DIR}/libmicroros/include"
)

pico_set_program_name(pico2w_led "pico2w_led")
pico_set_program_version(pico2w_led "0.2")

pico_enable_stdio_uart(pico2w_led 0)
pico_enable_stdio_usb(pico2w_led 1)

target_include_directories(pico2w_led PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Link Pico SDK, CYW43 Wi-Fi, hardware peripherals, and micro-ROS
target_link_libraries(pico2w_led
  pico_stdlib
  pico_cyw43_arch_lwip_threadsafe_background
  hardware_gpio
  hardware_i2c
  hardware_pwm
  hardware_sync
  micro_ros
)

pico_add_extra_outputs(pico2w_led)
