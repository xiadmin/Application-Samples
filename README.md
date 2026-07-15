# Application-Samples

Code samples for XIMEA cameras, showing how to control your camera using the XIMEA APIs. Each sample is self-contained and includes its own README.md file with detailed instructions.

---

## Prerequisites

- [XIMEA Software Package](https://www.ximea.com/software-downloads) installed (latest beta).
- Python 3.11 or newer for Python samples and maintenance scripts.
- CMake 3.16 or newer plus a C/C++ compiler for xiAPI C/C++ samples.
- .NET SDK 8.0 or newer for xiAPI.NET samples.

---

## Repository layout

```text
samples/
  xiapi/                    # C samples using xiAPI, plus xiAPI-based hardware-specific samples
  xiapiplus/                # C++ samples using xiAPIplus
  xiapi.net-c#/             # C# samples using xiAPI.NET
  xiapi-python/             # Python samples using xiAPI Python bindings
cmake/                      # shared/root CMake support files
scripts/                    # maintenance, generation, and build helpers
```

Each sample folder contains everything needed to build and run it, along with a README.md explaining what the sample does and how to use it.

---

## Available samples

| Sample | API / Language | What it shows |
|--------|-----------------|---------------|
| [xiapi/cross-platform/capture-10-images](samples/xiapi/cross-platform/capture-10-images/) | XiAPI (C) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapiplus/cross-platform/capture-10-images](samples/xiapiplus/cross-platform/capture-10-images/) | xiAPIplus (C++) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi.net-c#/capture-10-images](samples/xiapi.net-c#/capture-10-images/) | XiAPI.NET (C#) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-python/capture-10-images](samples/xiapi-python/capture-10-images/) | XiApiPython | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi/hardware-specific/jetson-demosaic-sample](samples/xiapi/hardware-specific/jetson-demosaic-sample/) | XiAPI (C++/CUDA) | GPU demosaicing on Nvidia Jetson, bypassing xiAPI CPU processing |
| [xiapi-python/gpio-samples-jetson](samples/xiapi-python/gpio-samples-jetson/) | XiApiPython | GPIO, I2C, SPI and UART samples for the Jetson kit |

---

## Building samples

Use the canonical full-repo build helper:

```bash
python3 scripts/build.py
```

For C/C++ samples only, you can also configure the root CMake entry directly:

```bash
cmake -S cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

Alternatively, build each sample separately by following the instructions in its own README.md file.

---

## Links

- [XIMEA API Manual](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
