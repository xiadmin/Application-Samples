# compute-histogram-cuda — C++ sample

Delivers 8-bit frames to CPU, CUDA managed, mapped host or GPU memory with XI_PRM_TRANSPORT_DATA_TARGET, computes a 64-bin histogram of each frame with a CUDA kernel and prints it with the processing time and throughput.

| Item | Value |
|------|-------|
| Category | GPU processing / CUDA integration |
| OS platform | Cross-platform |
| Hardware platform | NVIDIA GPU; GPUDirect RDMA target needs a XIMEA PCIe camera |
| API type | xiAPIplus |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+); `--target unified` and `--target gpu` need Linux |
| Hardware | Any XIMEA camera and an NVIDIA GPU; `--target gpu` needs a XIMEA PCIe camera and a GPU that supports GPUDirect RDMA |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| CUDA | [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11 or newer and the NVIDIA driver |
| CMake | 3.18 or newer |
| Compiler | MSVC 2022+ or GCC 9+ (C++17), supported by the installed CUDA Toolkit |
| Libraries | None besides xiAPIplus and the CUDA runtime |

When CMake does not find the CUDA Toolkit, the sample is skipped with a warning. With a Visual Studio generator, CMake also needs the toolkit's Visual Studio integration (selected in the CUDA installer). Without it, build with Ninja from a "x64 Native Tools Command Prompt for VS".

For `--target gpu`, enable GPUDirect support in the XIMEA PCIe driver first:

```bash
sudo /opt/XIMEA/src/ximea_cam_pcie/enable_gpudirect.sh
```

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/compute-histogram-cuda
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\compute-histogram-cuda
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

Without the CUDA Visual Studio integration, run these from an "x64 Native Tools Command Prompt for VS" instead:

```bat
cmake -B .cmake-tmp -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .cmake-tmp
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

```bash
# Linux
.cmake-tmp/build/xiapiplus-cross-platform-compute-histogram-cuda --target zerocopy

# Windows PowerShell
.\.cmake-tmp\build\xiapiplus-cross-platform-compute-histogram-cuda.exe --target cpu
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--target cpu` | yes | xiAPI writes frames to ordinary host memory; the sample copies each one to the GPU with `cudaMemcpy` |
| `--target unified` | | xiAPI writes frames to CUDA managed memory, which the kernel reads directly |
| `--target zerocopy` | | xiAPI writes frames to mapped (pinned) host memory, which the GPU reads over the bus |
| `--target gpu` | | The camera writes frames straight into GPU memory over GPUDirect RDMA |
| `--frame-count <n>` | 100 | Frames to process after one untimed warm-up frame |

Expected output:

```text
CUDA device 0: NVIDIA RTX A4000
Opening first camera
Image size 2064x1544, payload 3186816 bytes
Computing a 64-bin GPU histogram of 3186816 bytes for 100 frames
Frame 1/100 |     .',;;oOO%%88@@@@##@@88%%OOoo;;,'..                     | 0.412 ms
...
Average copy + histogram time: 0.405 ms, 7868.68 MB/s
Skipped frames: transport 0, API 0
Done
```

Each line draws the 64 bins from dark (few pixels) to bright (many pixels) on a logarithmic scale. The timing covers the copy to the GPU (`cpu` target only), the kernel and the read-back of the 64 counters. Compare the targets to see how much the copy costs.

---

## Known limitations / caveats

- **Data format:** xiAPI cannot post-process data it writes outside host memory, so every target uses `XI_FRM_TRANSPORT_DATA` with 8-bit output. That is MONO8 on monochrome cameras and RAW8 Bayer data on color cameras. The `cpu` target uses the same format so that the targets are comparable.
- **Unified memory:** `--target unified` needs `concurrentManagedAccess`, which CUDA does not offer on Windows.
- **Zero-copy:** on platforms where the GPU cannot use the host pointer directly, the sample translates it with `cudaHostGetDevicePointer`.
- **GPUDirect RDMA:** when the platform, driver or GPU does not support it, xiAPI returns `XI_GPUDIRECT_NOT_AVAILABLE` and the sample says so. The acquisition buffer is limited to 4 frames, because it lives in GPU memory exposed through the PCIe BAR, which is often only a few hundred MB.
- **GPU selection:** the sample always uses CUDA device 0.
- **Timing** uses the host clock around synchronized CUDA calls, so it includes launch overhead.
- The sample was ported from the internal `xiCUDASample`, whose build variants (pageable, bounce, unified, zero-copy, RDMA) became the `--target` option. The NVIDIA histogram kernel and helper headers were replaced by a short shared-memory `atomicAdd` kernel.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XI_PRM_TRANSPORT_DATA_TARGET](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_TRANSPORT_DATA_TARGET-or-transport_data_target)
- [GPUDirect RDMA with XIMEA cameras](https://www.ximea.com/support/wiki/apis/GPUDirect)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
