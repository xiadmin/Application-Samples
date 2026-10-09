# record-concatenated-images — MultiRecorder setup

Records 10 images in buffers of 2 concatenated images each and exports them as RGB TIFF files.

| Item | Value |
|------|-------|
| Category | Recording / concatenated images |
| OS platform | Cross-platform |
| Hardware platform | XIMEA PCIe xiX or xiB camera |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | One XIMEA PCIe camera of the xiX or xiB family |
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
samples\multirecorder\cross-platform\record-concatenated-images\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-concatenated-images/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there. MultiRecorder prints its progress and also writes it to `MultiRecorder-log.txt`.

The raw data goes to `data/`, and the 10 exported images go to `output/`.

---

## Known limitations / caveats

- **Architecture:** like record-10-images, but `concatImgCount="2"` makes the camera transfer two images in one buffer, which lowers the per-buffer overhead at high frame rates. The export splits the buffers back into single images.
- Only PCIe xiX and xiB cameras support concatenated images; other cameras report an error.
- Ported from the SP MultiRecorder sample `s7-concatenated-images`.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
