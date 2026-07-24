#!/usr/bin/env bash
set -euo pipefail

case "$(uname -m)" in
    x86_64)
        sdk_url="https://www.ximea.com/getattachment/1cbfaa8e-a175-4dab-badd-ab2961387799/XIMEA_macOX_SP.dmg"
        expected_arch="x86_64"
        ;;
    arm64)
        sdk_url="https://www.ximea.com/getattachment/8e005503-9914-4208-a80b-509dbbb3a901/XIMEA_macOS_ARM_SP.dmg"
        expected_arch="arm64"
        ;;
    *)
        echo "Unsupported macOS architecture: $(uname -m)" >&2
        exit 2
        ;;
esac

: "${RUNNER_TEMP:?RUNNER_TEMP must be set by GitHub Actions}"
: "${GITHUB_ENV:?GITHUB_ENV must be set by GitHub Actions}"

linux_support_url="https://www.ximea.com/getattachment/ab5baacf-e806-4b9d-b3d4-7eedf0f092b8/XIMEA_Linux_SP.tgz"
download_dir="$RUNNER_TEMP/ximea-download"
support_dir="$RUNNER_TEMP/ximea-portable-sources"
python_root="$RUNNER_TEMP/ximea-python"
sdk_root="$RUNNER_TEMP/ximea-sdk"
mount_dir="$RUNNER_TEMP/ximea-volume"
dmg="$download_dir/XIMEA_macOS_SP_latest_$(uname -m).dmg"
support_archive="$download_dir/XIMEA_Linux_SP_latest.tgz"

mkdir -p "$download_dir" "$support_dir" "$python_root" "$sdk_root/include/m3api" "$mount_dir"
curl --fail --location --retry 3 --silent --show-error "$sdk_url" --output "$dmg"
curl --fail --location --retry 3 --silent --show-error "$linux_support_url" --output "$support_archive"

hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" "$dmg"
trap 'hdiutil detach "$mount_dir" >/dev/null 2>&1 || true' EXIT
test -d "$mount_dir/m3api.framework"
test -d "$mount_dir/Examples/xiPython/v3/ximea"

sudo rm -rf /Library/Frameworks/m3api.framework
sudo ditto "$mount_dir/m3api.framework" /Library/Frameworks/m3api.framework
sudo xattr -dr com.apple.quarantine /Library/Frameworks/m3api.framework
codesign --verify --deep --strict /Library/Frameworks/m3api.framework
ditto "$mount_dir/Examples/xiPython/v3/ximea" "$python_root/ximea"

tar -xzf "$support_archive" -C "$support_dir"
portable_root="$support_dir/package"
if [ -f "$portable_root/version_LINUX_SP.txt" ]; then
    printf 'Downloaded XIMEA Linux portable sources %s\n' "$(tr -d '\r\n' < "$portable_root/version_LINUX_SP.txt")"
fi

cp "$portable_root/include"/*.h "$sdk_root/include/"
cp "$portable_root/include"/*.h "$sdk_root/include/m3api/"
cp /Library/Frameworks/m3api.framework/Headers/xiApi.h "$sdk_root/include/xiApi.h"
cp /Library/Frameworks/m3api.framework/Headers/xiApi.h "$sdk_root/include/m3api/xiApi.h"
cp -R "$portable_root/samples/_libs/xiAPIplus" "$sdk_root/xiAPIplus"
cp "$portable_root/samples/_libs/xiAPIplus/xiapiplus.h" "$sdk_root/include/xiApiPlus.h"
cp "$portable_root/samples/_libs/xiAPIplus/xiAPIplus_core.cpp" "$sdk_root/include/xiAPIplus_core.cpp"
cp "$portable_root/samples/_libs/xiAPIplus/xiAPIplus_parameters.cpp" "$sdk_root/include/xiAPIplus_parameters.cpp"
cp "$portable_root/samples/_libs/xiAPIplus/xiAPIplus_tiff.cpp" "$sdk_root/include/xiAPIplus_tiff.cpp"
cp "$portable_root/samples/_libs/xiAPIplus/xiAPIplus_tiff.h" "$sdk_root/include/xiAPIplus_tiff.h"
cp "$portable_root/samples/_libs/os_common_header.h" "$sdk_root/os_common_header.h"

framework_arches="$(lipo -archs /Library/Frameworks/m3api.framework/m3api)"
case " $framework_arches " in
    *" $expected_arch "*) ;;
    *) echo "m3api framework does not contain $expected_arch: $framework_arches" >&2; exit 1 ;;
esac

export XIMEA_ROOT="$sdk_root"
export XIMEA_SP_PATH="$sdk_root"
export PYTHONPATH="$python_root${PYTHONPATH:+:$PYTHONPATH}"
export DYLD_FRAMEWORK_PATH="/Library/Frameworks${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}"

{
    printf 'XIMEA_ROOT=%s\n' "$XIMEA_ROOT"
    printf 'XIMEA_SP_PATH=%s\n' "$XIMEA_SP_PATH"
    printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
    printf 'DYLD_FRAMEWORK_PATH=%s\n' "$DYLD_FRAMEWORK_PATH"
} >> "$GITHUB_ENV"

hdiutil detach "$mount_dir"
trap - EXIT
printf 'Installed latest XIMEA SDK beta for macOS %s at %s\n' "$(uname -m)" "$XIMEA_ROOT"
