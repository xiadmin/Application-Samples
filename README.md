# Application-Samples

Code samples for XIMEA cameras, showing how to control
your camera using the XIMEA APIs. Each sample is self-contained and includes
its own README.md file with detailed instructions.

---

## Prerequisites

- [XIMEA Software Package](https://www.ximea.com/software-downloads) installed (latest beta).

---

## Repository layout

    Samples/
      XiAPI/                    <- C/C++ samples using xiAPI
      XiAPI.NET-C#/             <- C# samples using xiAPI.NET
      XiApiPython/              <- Python samples using xiAPI Python bindings

Each sample folder contains everything needed to build and run it, along
with a README.md explaining what the sample does and how to use it.

---

## Available samples

| Sample | API / Language | What it shows |
|--------|-----------------|---------------|
| [XiAPI/Cross-Platform/Capture-10-images/c](Samples/XiAPI/Cross-Platform/Capture-10-images/c/) | XiAPI (C) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [XiAPI/Cross-Platform/Capture-10-images/cpp](Samples/XiAPI/Cross-Platform/Capture-10-images/cpp/) | XiAPI (C++) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [XiAPI.NET-C#/Capture-10-images](Samples/XiAPI.NET-C%23/Capture-10-images/) | XiAPI.NET (C#) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [XiApiPython/Capture-10-images](Samples/XiApiPython/Capture-10-images/) | XiApiPython | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [XiAPI/Hardware-specific/Jetson-Demosaic-Sample](Samples/XiAPI/Hardware-specific/Jetson-Demosaic-Sample/) | XiAPI (C++/CUDA) | GPU demosaicing on Nvidia Jetson, bypassing xiAPI CPU processing |
| [XiApiPython/GPIO samples-Jetson](<Samples/XiApiPython/GPIO samples-Jetson/>) | XiApiPython | GPIO, I2C, SPI and UART samples for the Jetson kit |

---

## Building samples

You may use `build.ps1` to build all samples directly.
Alternatively, you can build each sample separately by following the instructions in its own README.md file.

---

## Links

- [XIMEA API Manual](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
