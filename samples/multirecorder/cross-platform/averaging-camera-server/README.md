# averaging-camera-server — C++ sample

Runs a MultiRecorder camera server with a user image processor that averages every N frames on the camera computer, so that only the averaged images travel to the client.

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
| Hardware | Any XIMEA camera, connected to the computer that runs this server |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include the MultiRecorder sources |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+ or GCC 9+ (C++17) |
| Libraries | MultiRecorder sources from the SP (cpp-httplib and TinyXML included) |

CMake builds the camera server and client classes from `<XIMEA_SP_PATH>/Examples/MultiRecorder/sources` on Windows or `/opt/XIMEA/examples/MultiRecorder/sources` on Linux. When they are missing, the sample is skipped with a warning.

Run [averaging-camera-client](../averaging-camera-client/) on the same or another computer to receive the averaged images.

---

## Build

### Linux

```bash
cd samples/multirecorder/cross-platform/averaging-camera-server
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\multirecorder\cross-platform\averaging-camera-server
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

```bash
# Linux
.cmake-tmp/build/multirecorder-cross-platform-averaging-camera-server

# Windows PowerShell
.\.cmake-tmp\build\multirecorder-cross-platform-averaging-camera-server.exe --address 0.0.0.0
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--address <host>` | `localhost` | Address to listen at; `localhost` accepts local clients only, `0.0.0.0` or this computer's IP address accepts clients from other computers |
| `--port <port>` | 9001 | TCP port to listen at |

The server runs until you press Ctrl+C. Expected output while a client is connected:

```text
Starting camera server at localhost:9001; press Ctrl+C to stop
CameraServer started, listening at localhost:9001
List of connected devices:
idx: 0, name: MC031CG-SY-UB, type: USB30, sn: CECAU0000001, instPath: ...
Averaging set to 4 frames
```

---

## Known limitations / caveats

- **How it works:** `CxCameraServer` opens the camera when a client asks for it and handles every image request through `AveragingProcessor`, a `CxUserImageProcessor`. The client sets N with a user image processing request carrying `averaging="N"`. For N frames the server answers N-1 times with a text such as `Averaging frame 2/4`, then once with the averaged image.
- **Formats:** averaging works on `XI_MONO8`, `XI_RAW8` and `XI_RGB24`. With `averaging="1"` every frame passes through unchanged, in any format.
- A change of image size or format restarts the average.
- The server exits with an error when no camera is connected at start-up, and when it cannot listen at the address and port.
- The server has no authentication; expose it only on networks you trust, and allow the port through the firewall when clients run on other computers.
- Ported from the SP example `MultiRecorder_CameraSideProcessing` (`uipa_server_main.cpp`), with the address and port as arguments.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
