# capture-10-images — C sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.32+ |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |

---

## Build

Build from the sample folder using CMake directly, or use the PowerShell
helper at the repo root which builds all samples in one shot.

### CMake directly — Linux

```bash
cd samples/acquisition/capture-10-images/c
cmake -B .cmake-tmp 
cmake --build .cmake-tmp
```

Binary lands in `.cmake-tmp/build/`.

### CMake directly — Windows (PowerShell)

```powershell
cd samples\acquisition\capture-10-images\c
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

Binary lands in `.cmake-tmp\build\Release\`.

## Run

After a direct CMake build:

```bash
# Linux
.cmake-tmp/build/acquisition-capture-10-images-c

# Windows PowerShell
.\.cmake-tmp\build\Release\acquisition-capture-10-images-c.exe
```

---

## Expected output

```
Found 1 camera(s), opening index 0
Exposure: 100000 us (100 ms)
Capturing 10 frames
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

-

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
