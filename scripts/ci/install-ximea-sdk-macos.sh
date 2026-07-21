#!/usr/bin/env bash
set -euo pipefail

case "$(uname -m)" in
    x86_64)
        sdk_url="https://www.ximea.com/getattachment/e2ee7c15-fa5e-4e1b-b85f-5af6845bada3/XIMEA_macOX_SP.dmg"
        sdk_sha256="82d9971baf55592ad6c88a98788723b10ee301a678fdaeb4aae48a1006bd5543"
        expected_arch="x86_64"
        ;;
    arm64)
        sdk_url="https://www.ximea.com/getattachment/6ea47896-cfc1-4c33-a2cc-5dc7ee38de39/XIMEA_macOS_ARM_SP.dmg"
        sdk_sha256="b82cb968a0febc8e929f2e69be4c15a0d2835fb11182e42f1748a7c455779b31"
        expected_arch="arm64"
        ;;
    *)
        echo "Unsupported macOS architecture: $(uname -m)" >&2
        exit 2
        ;;
esac

: "${RUNNER_TEMP:?RUNNER_TEMP must be set by GitHub Actions}"
: "${GITHUB_ENV:?GITHUB_ENV must be set by GitHub Actions}"

linux_support_url="https://www.ximea.com/getattachment/281fd5c5-3335-4279-a494-f49c004f00c6/XIMEA_Linux_SP.tgz"
linux_support_sha256="b3d3da3d5e3f0417a54788726110103a03738dbddbb70703026bc029d6044ffa"
download_dir="$RUNNER_TEMP/ximea-download"
support_dir="$RUNNER_TEMP/ximea-portable-sources"
python_root="$RUNNER_TEMP/ximea-python"
sdk_root="$RUNNER_TEMP/ximea-sdk"
mount_dir="$RUNNER_TEMP/ximea-volume"
dmg="$download_dir/XIMEA_macOS_SP_V4.32.00_$(uname -m).dmg"
support_archive="$download_dir/XIMEA_Linux_SP_V4.32.00.tgz"

mkdir -p "$download_dir" "$support_dir" "$python_root" "$sdk_root/include/m3api" "$mount_dir"
curl --fail --location --retry 3 --silent --show-error "$sdk_url" --output "$dmg"
printf '%s  %s\n' "$sdk_sha256" "$dmg" | shasum --algorithm 256 --check
curl --fail --location --retry 3 --silent --show-error "$linux_support_url" --output "$support_archive"
printf '%s  %s\n' "$linux_support_sha256" "$support_archive" | shasum --algorithm 256 --check

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
test "$(tr -d '\r\n' < "$portable_root/version_LINUX_SP.txt")" = "LINUX_SP_V4_32_00"

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
export PYTHONPATH="$python_root${PYTHONPATH:+:$PYTHONPATH}"
export DYLD_FRAMEWORK_PATH="/Library/Frameworks${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}"

{
    printf 'XIMEA_ROOT=%s\n' "$XIMEA_ROOT"
    printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
    printf 'DYLD_FRAMEWORK_PATH=%s\n' "$DYLD_FRAMEWORK_PATH"
} >> "$GITHUB_ENV"

hdiutil detach "$mount_dir"
trap - EXIT
printf 'Installed XIMEA SDK LTS V4.32.00 for macOS %s at %s\n' "$(uname -m)" "$XIMEA_ROOT"
