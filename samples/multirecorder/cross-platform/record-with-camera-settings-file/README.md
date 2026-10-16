# record-with-camera-settings-file — MultiRecorder setup

Applies camera settings saved by xiCamTool from `camera-settings.xicamera`, overrides the exposure, records 10 images and exports them as RGB TIFF files.

| Item | Value |
|------|-------|
| Category | Recording / camera settings file |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | One XIMEA camera; the included settings file was saved from an MC031CG-SY-UB |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include MultiRecorder |
| Libraries | None |

The scripts run MultiRecorder from `%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe` on Windows and `/opt/XIMEA/examples/MultiRecorder/MultiRecorder` on Linux. Set the `MULTIRECORDER` environment variable to the full path of the executable to use another copy.

---

## Build

No build step. `setup.xml` describes the recording and refers to `camera-settings.xicamera`; `run.bat` and `run.sh` pass it to MultiRecorder.

---

## Run

```bat
:: Windows
samples\multirecorder\cross-platform\record-with-camera-settings-file\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-with-camera-settings-file/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there.

---

## Expected output

MultiRecorder prints its progress to the console and also writes it to `MultiRecorder-log.txt`. After a successful run the sample folder contains:

```text
samples/multirecorder/cross-platform/record-with-camera-settings-file/
  camera-settings.xicamera   settings file (part of the sample, not created by the run)
  data/                      raw camera data written during the acquisition
  output/                    10 exported RGB (XI_RGB24) TIFF images
  MultiRecorder-log.txt      MultiRecorder log
```

Settings from the file that the camera does not support are skipped without an error. MultiRecorder exits with code 0 even when a task fails, so confirm that `output/` holds 10 TIFF files and that the console output or `MultiRecorder-log.txt` reports no errors.

---

## Known limitations / caveats

- **Architecture:** like record-10-images, but before recording MultiRecorder applies every parameter stored in the `.xicamera` file, then the `setCameraParam` elements that follow it (here a 1 ms exposure).
- The settings file is specific to the camera model it was saved from. Parameters that another model does not support, or values outside its range (such as the 2064x1544 resolution), are skipped without an error. Save your own file in xiCamTool with the menu of the (i) toolbar button and replace `camera-settings.xicamera`.
- The serial-number attributes of the original file were removed; MultiRecorder reads only the `<Values>` element.
- Ported from the SP MultiRecorder sample `s2-use-camera-settings`.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
