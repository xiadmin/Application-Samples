# export-recorded-session — MultiRecorder setup

Records 10 images into a saved recording session, then exports the session to TIFF files in a second MultiRecorder run.

| Item | Value |
|------|-------|
| Category | Recording / post-processing |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | One XIMEA camera |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include MultiRecorder |
| Libraries | None |

The scripts run MultiRecorder from `%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe` on Windows and `/opt/XIMEA/examples/MultiRecorder/MultiRecorder` on Linux. Set the `MULTIRECORDER` environment variable to the full path of the executable to use another copy.

---

## Build

No build step. `setup.xml` describes the recording; `run.bat` and `run.sh` run it and then export the saved session.

---

## Run

```bat
:: Windows
samples\multirecorder\cross-platform\export-recorded-session\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/export-recorded-session/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there.

---

## Expected output

MultiRecorder runs twice: first to record, then to export. Both runs print their progress to the console, and MultiRecorder also writes it to `MultiRecorder-log.txt`. After a successful run the sample folder contains:

```text
samples/multirecorder/cross-platform/export-recorded-session/
  data/                          raw camera data of the 10 recorded images
  recorded-session.xirecsession  saved session description
  exported/                      10 exported RGB (XI_RGB24) TIFF images
  MultiRecorder-log.txt          MultiRecorder log
```

If the recording fails, the export reports that it cannot open the session file. MultiRecorder exits with code 0 even when a task fails, so confirm that `exported/` holds 10 TIFF files and that the console output or `MultiRecorder-log.txt` reports no errors.

---

## Known limitations / caveats

- **Architecture:** recording and conversion are separate steps. The first run keeps the raw data in the storage and saves the session (`saveSession`); the second (`--export-tiffs`) converts it later, for example when the acquisition computer is idle. A saved session can also be opened in xiCamTool.
- `exportImageFormat` in `setup.xml` sets the image format the export uses.
- If the recording fails, the export reports that it cannot open the session file.
- Ported from the SP MultiRecorder sample `s6-post_process`.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
