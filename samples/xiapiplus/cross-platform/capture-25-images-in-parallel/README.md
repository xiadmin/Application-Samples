# capture-25-images-in-parallel — C++ sample

Opens all connected cameras in parallel, triggers a master camera by software and the slaves through its GPO1 exposure pulse, and captures 5 bursts of 5 frames per camera on per-camera acquisition threads with a shared worker pool.

| Item | Value |
|------|-------|
| Category | Multi-camera / hardware trigger |
| OS platform | Cross-platform |
| Hardware platform | Two or more XIMEA cameras with GPIO, master GPO1 wired to slave GPI1 |
| API type | xiAPIplus |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Two or more XIMEA cameras with GPIO; one acts as master, the others as slaves |
| Wiring | Master GPO1 connected to GPI1 of every slave (plus a common ground); see the camera's GPIO documentation for pinout and voltage levels |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | None |

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/capture-25-images-in-parallel
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-25-images-in-parallel
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

From the repository root (the first camera found acts as master):

```bash
# Linux
cd samples/xiapiplus/cross-platform/capture-25-images-in-parallel
.cmake-tmp/build/xiapiplus-cross-platform-capture-25-images-in-parallel

# Windows PowerShell
cd samples\xiapiplus\cross-platform\capture-25-images-in-parallel
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-25-images-in-parallel.exe
```

Optional: choose the master camera explicitly. Replace `<serial>` with the serial number of a connected camera before running:

```bash
# Linux
.cmake-tmp/build/xiapiplus-cross-platform-capture-25-images-in-parallel --master <serial>

# Windows PowerShell
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-25-images-in-parallel.exe --master <serial>
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--master <serial>` | first camera found | Serial number of the camera whose GPO1 drives the others |
| `--bursts <n>` | 5 | Software triggers sent to the master, one per second |
| `--burst-length <n>` | 5 | Frames per trigger on every camera |
| `--total-bandwidth <Mbps>` | 19200 | Host bandwidth, split equally between the cameras |

---

## Expected output

```text
  [0] XECAS1930001 (master)
  [1] XECAS1930002 (slave)
Opened 2 cameras in 850 ms
Bandwidth limit per camera: 9600 Mbps
Configured camera XECAS1930001 as master
Configured camera XECAS1930002 as slave
--- burst 1/5 ---
camera XECAS1930001 [0] frame 1 t=0.012 s 1280x1024 first pixel 37
camera XECAS1930002 [1] frame 1 t=0.012 s 1280x1024 first pixel 41
...
Camera XECAS1930001: grabbed 25 of 25, skipped on transport 0
Camera XECAS1930002: grabbed 25 of 25, skipped on transport 0
Done
```

Lines from different cameras interleave in arrival order. The sample exits with a failure code when any camera grabbed fewer frames than expected or skipped frames on transport. That usually means missing wiring or too little bandwidth.

---

## Known limitations / caveats

- **Synchronization:** the master runs with `XI_TRG_SEL_FRAME_BURST_START` and a software trigger, and pulses GPO1 (`XI_GPO_EXPOSURE_PULSE`) at the start of every exposure. Each slave starts one frame per rising edge on GPI1 (`XI_TRG_SEL_FRAME_START`, `XI_TRG_EDGE_RISING`). Slaves are started 50 ms before the master so that they do not miss the first pulse.
- **Bandwidth:** `XI_PRM_AUTO_BANDWIDTH_CALCULATION` is switched off globally before any camera is opened, and every camera is limited to `total-bandwidth / camera count`. Lower `--total-bandwidth` if frames are skipped on transport.
- **Memory:** every camera gets a 512 MiB acquisition buffer. With many cameras, check that the host has enough RAM.
- **Timestamps and overlap:** trigger overlap and the software timestamp reset are optional. On cameras that do not support them the sample prints a warning and continues.
- **Format:** frames are captured as RAW8 so that the sample runs on any model. The legacy version configured 12-bit packed RAW16, which only some sensors support.
- **Single camera:** with only one camera the sample still runs; it acts as master without slaves.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [xiAPI trigger and GPIO parameters](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_TRG_SOURCE)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
