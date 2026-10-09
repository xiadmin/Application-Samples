# record-10-images — MultiRecorder setup

Records 10 images from the first connected camera with a 200 ms exposure and fixed white balance, then exports them as RGB TIFF files.

| Item | Value |
|------|-------|
| Category | Recording / basic acquisition |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | One XIMEA camera; a color camera gives meaningful RGB output |
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
samples\multirecorder\cross-platform\record-10-images\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-10-images/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there. MultiRecorder prints its progress and also writes it to `MultiRecorder-log.txt`.

The raw data goes to `data/`, and the 10 exported images go to `output/`.

---

## Known limitations / caveats

- **Architecture:** a single MultiRecorder process opens the camera, writes the raw transport data to the storage folder during the acquisition and converts it to TIFF afterwards. Writing raw data first keeps up with high frame rates.
- The camera parameters are set by name (`XI_PRM_EXPOSURE` or `exposure`); enum values can be given by name (`XI_DWN_1x1`).
- Ported from the SP MultiRecorder sample `s1-capture-10-images`.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
