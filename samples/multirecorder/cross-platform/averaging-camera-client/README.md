# averaging-camera-client — C++ sample

Connects to an averaging camera server, configures its camera, asks the server to average N frames per image and prints the size, frame number and mean value of every averaged image it receives.

| Item | Value |
|------|-------|
| Category | Client-server / camera-side processing |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPI + MultiRecorder |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | None on the client computer; the camera is connected to the server computer |
| Server | [averaging-camera-server](../averaging-camera-server/) running and reachable over the network |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include the MultiRecorder sources |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+ or GCC 9+ (C++17) |
| Libraries | MultiRecorder sources from the SP (cpp-httplib and TinyXML included) |

CMake builds the camera client classes from `<XIMEA_SP_PATH>/Examples/MultiRecorder/sources` on Windows or `/opt/XIMEA/examples/MultiRecorder/sources` on Linux. When they are missing, the sample is skipped with a warning.

---

## Build

### Linux

```bash
cd samples/multirecorder/cross-platform/averaging-camera-client
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\multirecorder\cross-platform\averaging-camera-client
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

Start the server first, then, from the repository root:

```bash
# Linux
cd samples/multirecorder/cross-platform/averaging-camera-client
.cmake-tmp/build/multirecorder-cross-platform-averaging-camera-client --averaging 8

# Windows PowerShell
cd samples\multirecorder\cross-platform\averaging-camera-client
.\.cmake-tmp\build\multirecorder-cross-platform-averaging-camera-client.exe --server 192.168.1.10:9001
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--server <host:port>` | `localhost:9001` | Camera server to connect to |
| `--averaging <n>` | 4 | Frames the server averages into one image |
| `--frame-count <n>` | 100 | Camera frames to request; the client receives `frame-count / averaging` images |
| `--exposure-us <us>` | 10000 | Exposure time in microseconds |

---

## Expected output

```text
Opening camera MC031CG-SY-UB (CECAU0000001) at localhost:9001
Server averages 4 frames per image
Server: Averaging frame 1/4
Server: Averaging frame 2/4
Server: Averaging frame 3/4
Averaged image 1: 2064x1544, last frame number 4, mean value 87.3
...
Done
```

---

## Known limitations / caveats

- The client opens the first camera the server reports.
- Color cameras deliver `XI_RGB24` with automatic white balance, monochrome cameras `XI_MONO8`, because the server averages 8-bit data only.
- Instead of the OpenCV preview window of the original example, the client prints statistics of every image, so it runs without OpenCV and without a display.
- When the server cannot be reached, the client exits with "could not connect to the camera server". The connection attempt times out after 5 seconds.
- Ported from the SP example `MultiRecorder_CameraSideProcessing` (`uipa_client_main.cpp`), with the server address, averaging, frame count and exposure as arguments.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
