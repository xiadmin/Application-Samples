# record-through-local-server — MultiRecorder setup

Starts a MultiRecorder camera server on this computer, records 10 images from its camera through the server, then stops the server.

| Item | Value |
|------|-------|
| Category | Recording / client-server |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | One XIMEA camera; port 8090 free on this computer |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include MultiRecorder |
| Libraries | None |

The scripts run MultiRecorder from `%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe` on Windows and `/opt/XIMEA/examples/MultiRecorder/MultiRecorder` on Linux. Set the `MULTIRECORDER` environment variable to the full path of the executable to use another copy.

---

## Build

No build step. `setup.xml` describes the recording; `run.bat` and `run.sh` start the camera server, pass the setup to a second MultiRecorder process and stop the server.

---

## Run

```bat
:: Windows
samples\multirecorder\cross-platform\record-through-local-server\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-through-local-server/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there. MultiRecorder prints its progress and also writes it to `MultiRecorder-log.txt`.

The server writes the raw data to `data/` and the exported images to `output/`. Its own log goes to `server-log.txt`.

---

## Known limitations / caveats

- **Architecture:** two MultiRecorder processes. The camera server (`--start-server localhost:8090`) owns the camera and the storage; the recording process connects to it (`connectToServer`) and controls the task. Splitting the work this way spreads several cameras over several CPU cores or computers.
- On Windows the server runs in its own minimized console window titled `multirecorder-sample-server`, and `run.bat` closes it by that title. On Linux it runs in the background and `run.sh` stops it on exit.
- Ported from the SP MultiRecorder sample `s4-server-multi-cpu`. The original Windows script stopped every running `MultiRecorder.exe`; this one stops only the server it started.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
