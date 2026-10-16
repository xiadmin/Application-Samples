# record-through-remote-server — MultiRecorder setup

Records 10 images from a camera connected to another computer, which runs a MultiRecorder camera server.

| Item | Value |
|------|-------|
| Category | Recording / client-server |
| OS platform | Cross-platform |
| Hardware platform | A camera computer and a recording computer on one network |
| API type | xiAPI + MultiRecorder |
| Language | MultiRecorder recording setup (XML) |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | One XIMEA camera on the server computer, and a network connection from the recording computer to the server's port 8090 |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) with its examples, which include MultiRecorder |
| Libraries | None |

The scripts run MultiRecorder from `%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe` on Windows and `/opt/XIMEA/examples/MultiRecorder/MultiRecorder` on Linux. Set the `MULTIRECORDER` environment variable to the full path of the executable to use another copy.

---

## Build

No build step. `start-server.bat`/`start-server.sh` run on the camera computer. `setup.xml` describes the recording; `run.bat` and `run.sh` pass it to MultiRecorder on the recording computer.

---

## Run

On the computer the camera is connected to, start the server and leave it running (Ctrl+C stops it). Pass its IP address and port, or nothing for `localhost:8090`:

```bat
:: Windows
samples\multirecorder\cross-platform\record-through-remote-server\start-server.bat 192.168.1.10:8090
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-through-remote-server/start-server.sh 192.168.1.10:8090
```

On the recording computer, set the same address in `setup.xml`, then run:

```bat
:: Windows
samples\multirecorder\cross-platform\record-through-remote-server\run.bat
```

```bash
# Linux
sh samples/multirecorder/cross-platform/record-through-remote-server/run.sh
```

The scripts change to the sample folder first, so the relative folders and files in `setup.xml` are created there.

---

## Expected output

The recording process prints its progress to the console and also writes it to `MultiRecorder-log.txt` on the recording computer. The server keeps running after the recording until you stop it with Ctrl+C.

The data and the exported images stay on the server computer, in the sample folder next to `start-server`:

```text
samples/multirecorder/cross-platform/record-through-remote-server/   (server computer)
  data/                   raw camera data
  output/                 10 exported RGB (XI_RGB24) TIFF images
```

MultiRecorder exits with code 0 even when a task fails, so confirm that `output/` on the server computer holds 10 TIFF files and that the console output or `MultiRecorder-log.txt` reports no errors.

---

## Known limitations / caveats

- **Architecture:** like record-through-local-server, but the camera server and the recording process run on different computers. Only control commands travel over the network; the image data is written on the camera computer.
- The default address `localhost:8090` lets you try both parts on one computer. For two computers, start the server with the camera computer's IP address and put the same address in the `connectToServer` attribute of `setup.xml`, and allow the port through the firewall.
- The camera server has no authentication; use it only on networks you trust.
- Ported from the SP MultiRecorder sample `s5-server-remote`.
- MultiRecorder exits with code 0 even when a task fails, so check the console output or `MultiRecorder-log.txt`.
- The `.gitignore` in this folder's parent keeps the recorded data, exported images and logs out of git.

---

## Links

- [MultiRecorder](https://www.ximea.com/support/wiki/allprod/MultiRecorder)
- [xiAPI parameters](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
