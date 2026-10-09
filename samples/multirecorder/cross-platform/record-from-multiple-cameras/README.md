# record-from-multiple-cameras — MultiRecorder setup

Records 10 images from each of three cameras connected to this computer, with different settings, storages and export formats per camera.

| Item | Value |
|------|-------|
| Category | Recording / multi-camera |
| OS platform | Cross-platform |
| Hardware platform | Three XIMEA cameras connected to one computer |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Three XIMEA cameras connected to this computer; camera 0 should be a color camera |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include MultiRecorder |
| Libraries | None |

The scripts run MultiRecorder from `%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe` on Windows and `/opt/XIMEA/examples/MultiRecorder/MultiRecorder` on Linux. Set the `MULTIRECORDER` environment variable to the full path of the executable to use another copy.

---

## Build

No build step. `setup.xml` describes the recording; `run.bat` and `run.sh` pass it to MultiRecorder.

---

## Run

```bat
:: Windows
samples\multirecorder\cross-platform\record-from-multiple-cameras\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-from-multiple-cameras/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there. MultiRecorder prints its progress and also writes it to `MultiRecorder-log.txt`.

Cameras 0 and 1 share storage `s1`, whose frames alternate between `data-1/` and `data-2/`; camera 2 writes to `data-3/`. The exports go to `output-camera-1/` (RGB32), `output-camera-2/` and `output-camera-3/` (MONO16).

---

## Known limitations / caveats

- **Architecture:** one MultiRecorder process runs one camera task per camera in parallel. A storage with several folders spreads the frames over them, which is faster when the folders are on different drives.
- Cameras are selected by index (`cameraIdx`). Use `cameraSn`, `cameraUserID`, `cameraInstPath` or `cameraLocPath` to select them independently of the enumeration order.
- With fewer than three cameras, the tasks for the missing ones fail and the others still record.
- Ported from the SP MultiRecorder sample `s3-multi-cameras`.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
