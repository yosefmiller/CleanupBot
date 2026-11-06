#!/usr/bin/env bash

set -euo pipefail

# 1. Resolve source and output paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PICO2W_DIR="$(cd "${SCRIPT_DIR}/../pico2w_led" && pwd)"
OUTPUT_DIR="${PICO2W_DIR}/libmicroros"

# 2. Use a staging workspace inside Linux (/tmp)
BUILD_DIR="/tmp/microros_build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/src" "${BUILD_DIR}/bin"

echo "=== Staging micro-ROS build in temporary directory: ${BUILD_DIR} ==="

# Copy packages and configs into staging area
cp -r "${SCRIPT_DIR}/src/micro_ros_setup" "${BUILD_DIR}/src/"
cp "${SCRIPT_DIR}/toolchain.cmake" "${BUILD_DIR}/"
cp "${SCRIPT_DIR}/colcon.meta" "${BUILD_DIR}/"

# Point toolchain and SDK to local Pico tools via symlinks in /tmp
ln -s "${PICO2W_DIR}/.tools/pico-sdk" "${BUILD_DIR}/pico-sdk"
ln -s "${PICO2W_DIR}/.tools/arm-none-eabi" "${BUILD_DIR}/arm-none-eabi"
export PICO_SDK_PATH="${BUILD_DIR}/pico-sdk"

# Symlink all Pixi build tools into /tmp/microros_build/bin
PIXI_BIN="$(dirname "$(which colcon)")"
for f in "${PIXI_BIN}"/*; do
  name="$(basename "$f")"
  if [[ ! "$name" =~ ^(ros|ament|rcl|rviz) ]]; then
    ln -sf "$f" "${BUILD_DIR}/bin/$name" 2>/dev/null || true
  fi
done

# Ensure standard un-prefixed compiler names (gcc, g++, cc, c++) exist in staging
HOST_GCC=$(find "${PIXI_BIN}" /usr/bin -name '*linux-gnu-gcc' -o -name 'gcc' 2>/dev/null | head -n1)
HOST_GXX=$(find "${PIXI_BIN}" /usr/bin -name '*linux-gnu-g++' -o -name 'g++' 2>/dev/null | head -n1)
HOST_AR=$(find "${PIXI_BIN}" /usr/bin -name '*linux-gnu-ar' -o -name 'ar' 2>/dev/null | head -n1)
HOST_RANLIB=$(find "${PIXI_BIN}" /usr/bin -name '*linux-gnu-ranlib' -o -name 'ranlib' 2>/dev/null | head -n1)

if [ -n "$HOST_GCC" ]; then
  ln -sf "$HOST_GCC" "${BUILD_DIR}/bin/gcc"
  ln -sf "$HOST_GCC" "${BUILD_DIR}/bin/cc"
fi
if [ -n "$HOST_GXX" ]; then
  ln -sf "$HOST_GXX" "${BUILD_DIR}/bin/g++"
  ln -sf "$HOST_GXX" "${BUILD_DIR}/bin/c++"
fi
if [ -n "$HOST_AR" ]; then
  ln -sf "$HOST_AR" "${BUILD_DIR}/bin/ar"
fi
if [ -n "$HOST_RANLIB" ]; then
  ln -sf "$HOST_RANLIB" "${BUILD_DIR}/bin/ranlib"
fi

# Ensure make, python, etc. are linked
for tool in make ninja python python3 git pkg-config tar; do
  tool_path=$(which "$tool" 2>/dev/null || true)
  if [ -n "$tool_path" ] && [ ! -e "${BUILD_DIR}/bin/$tool" ]; then
    ln -sf "$tool_path" "${BUILD_DIR}/bin/$tool"
  fi
done

# Add a rosdep shim (Pixi manages dependencies, so rosdep is a no-op)
cat << 'SHIM' > "${BUILD_DIR}/bin/rosdep"
#!/usr/bin/env bash
exit 0
SHIM
chmod +x "${BUILD_DIR}/bin/rosdep"

# Ensure our staging bin and ARM toolchain are on PATH
export PATH="${BUILD_DIR}/bin:${BUILD_DIR}/arm-none-eabi/bin:${PATH}"

cd "${BUILD_DIR}"

# 3. Build micro_ros_setup in the clean path
colcon build --packages-select micro_ros_setup
set +u
source "${BUILD_DIR}/install/setup.bash"
set -u

# 4. Download micro-ROS sources and cross-compile
echo "=== Downloading micro-ROS client sources ==="
ros2 run micro_ros_setup create_firmware_ws.sh generate_lib

echo "=== Cross-compiling for Cortex-M33 (RP2350) ==="
ros2 run micro_ros_setup build_firmware.sh \
    "${BUILD_DIR}/toolchain.cmake" \
    "${BUILD_DIR}/colcon.meta"

ARCHIVE="${BUILD_DIR}/firmware/build/libmicroros.a"
if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Expected archive was not produced: ${ARCHIVE}" >&2
    exit 1
fi

# 5. Verify Cortex-M33 architecture
CHECK_DIR="${BUILD_DIR}/archive-check"
mkdir -p "${CHECK_DIR}"
cd "${CHECK_DIR}"
arm-none-eabi-ar x "${ARCHIVE}"
find . -type f \( -name '*.o' -o -name '*.obj' \) \
    -exec arm-none-eabi-readelf -A {} \; > attributes.txt 2>/dev/null

if grep -Eq 'v6-M|6-M|Cortex-M0' attributes.txt; then
    echo "Archive contains RP2040/Cortex-M0+ objects; refusing to publish it." >&2
    exit 1
fi

if ! grep -Eqi 'v8-M\.main(line)?|8-M\.MAIN|Cortex-M33' attributes.txt; then
    echo "Could not confirm ARMv8-M Mainline/Cortex-M33 objects." >&2
    exit 1
fi

# 6. Copy final outputs directly back to project folder
echo "=== Publishing outputs to ${OUTPUT_DIR} ==="
mkdir -p "${OUTPUT_DIR}"
cp -R "${BUILD_DIR}/firmware/build/include" "${OUTPUT_DIR}/"
cp "${ARCHIVE}" "${OUTPUT_DIR}/libmicroros.a"
cp attributes.txt "${OUTPUT_DIR}/architecture.txt"

# Clean up staging directory
rm -rf "${BUILD_DIR}"

echo "Successfully created ${OUTPUT_DIR}/libmicroros.a"
grep -E 'Tag_CPU_(name|arch)' "${OUTPUT_DIR}/architecture.txt" | sort -u