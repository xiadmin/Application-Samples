# error-handling — Python sample

Shows how to detect and recover from an xiAPI error such as XI_TIMEOUT.

| Item | Value |
|------|-------|
| Category | Error handling |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI Python |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Build

No build step is required for this Python sample.

---

## Run

```bash
cd samples/xiapi-python/cross-platform/error-handling
python main.py
```

---

## Expected output

The sample first calls the API without an open camera, which fails with `XI_INVALID_HANDLE` and needs no hardware. It then opens the camera with a 100 ms exposure and waits only 1 ms for the first frame. The resulting `XI_TIMEOUT` is caught and handled by retrying with a 5000 ms timeout. Any other error ends the sample with a non-zero exit code; on success it exits with 0.

```text
Caught XI_INVALID_HANDLE (1) from a call without an open camera
Opening camera index 0
Caught XI_TIMEOUT (10) with a 1 ms timeout; retrying with 5000 ms
Got frame 1280x1024 nframe=1
Done
```

Without a camera, only the first line is printed, followed by `Error: no XIMEA cameras detected` on stderr and exit code 1.

---

## Known limitations / caveats

- The `ximea` package raises `xiapi.Xi_error` with the code in `status`, but does not name the codes; the sample defines `XI_INVALID_HANDLE` and `XI_TIMEOUT` from `xiApi.h`.
- The `XI_TIMEOUT` path depends on the first frame taking longer than 1 ms. If a frame does arrive in time, the sample prints `A frame arrived within 1 ms; no timeout this time` and continues.
- The sample always opens camera index 0.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
