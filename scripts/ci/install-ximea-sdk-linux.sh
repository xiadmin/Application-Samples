#!/usr/bin/env bash
set -euo pipefail

case "$(uname -m)" in
    x86_64)
        sdk_url="https://www.ximea.com/getattachment/ab5baacf-e806-4b9d-b3d4-7eedf0f092b8/XIMEA_Linux_SP.tgz"
        sdk_arch="X64"
        ;;
    aarch64|arm64)
        sdk_url="https://www.ximea.com/getattachment/21790810-a3ed-4183-948f-e999cfcf8233/XIMEA_Linux_ARM_SP.tgz"
        sdk_arch="Xarm64"
        ;;
    *)
        echo "Unsupported Linux architecture: $(uname -m)" >&2
        exit 2
        ;;
esac

: "${RUNNER_TEMP:?RUNNER_TEMP must be set by GitHub Actions}"
: "${GITHUB_ENV:?GITHUB_ENV must be set by GitHub Actions}"

download_dir="$RUNNER_TEMP/ximea-download"
extract_dir="$RUNNER_TEMP/ximea-package"
sdk_root="$RUNNER_TEMP/ximea-sdk"
archive="$download_dir/XIMEA_Linux_SP_latest_$(uname -m).tgz"

mkdir -p "$download_dir" "$extract_dir" "$sdk_root/include/m3api" "$sdk_root/lib"
curl --fail --location --retry 3 --silent --show-error "$sdk_url" --output "$archive"
tar -xzf "$archive" -C "$extract_dir"

package_root="$extract_dir/package"
if [ -f "$package_root/version_LINUX_SP.txt" ]; then
    printf 'Downloaded XIMEA Linux SDK %s\n' "$(tr -d '\r\n' < "$package_root/version_LINUX_SP.txt")"
fi

sudo apt-get update
sudo apt-get install --yes libraw1394-11 libtiff6 libusb-1.0-0

sudo install -m 0644 "$package_root/api/$sdk_arch/libm3api.so.2" /usr/lib/libm3api.so.2
sudo ln -sfn libm3api.so.2 /usr/lib/libm3api.so

libtiff6="$(ldconfig -p | sed -n 's/.*libtiff\.so\.6.* => //p' | head -n 1)"
test -n "$libtiff6"
sudo ln -sfn "$libtiff6" /usr/lib/libtiff.so.5
sudo ldconfig

cp "$package_root/include"/*.h "$sdk_root/include/"
cp "$package_root/include"/*.h "$sdk_root/include/m3api/"
cp -R "$package_root/samples/_libs/xiAPIplus" "$sdk_root/xiAPIplus"
cp "$package_root/samples/_libs/xiAPIplus/xiapiplus.h" "$sdk_root/include/xiApiPlus.h"
cp "$package_root/samples/_libs/xiAPIplus/xiAPIplus_core.cpp" "$sdk_root/include/xiAPIplus_core.cpp"
cp "$package_root/samples/_libs/xiAPIplus/xiAPIplus_parameters.cpp" "$sdk_root/include/xiAPIplus_parameters.cpp"
cp "$package_root/samples/_libs/xiAPIplus/xiAPIplus_tiff.cpp" "$sdk_root/include/xiAPIplus_tiff.cpp"
cp "$package_root/samples/_libs/xiAPIplus/xiAPIplus_tiff.h" "$sdk_root/include/xiAPIplus_tiff.h"
cp "$package_root/samples/_libs/os_common_header.h" "$sdk_root/os_common_header.h"
cp "$package_root/api/$sdk_arch/libm3api.so.2" "$sdk_root/lib/libm3api.so.2"
ln -sfn libm3api.so.2 "$sdk_root/lib/libm3api.so"

export XIMEA_ROOT="$sdk_root"
export XIMEA_SP_PATH="$sdk_root"
export PYTHONPATH="$package_root/api/Python/v3${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$sdk_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
    printf 'XIMEA_ROOT=%s\n' "$XIMEA_ROOT"
    printf 'XIMEA_SP_PATH=%s\n' "$XIMEA_SP_PATH"
    printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
    printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
} >> "$GITHUB_ENV"

printf 'Installed latest XIMEA SDK beta for Linux %s at %s\n' "$(uname -m)" "$XIMEA_ROOT"
