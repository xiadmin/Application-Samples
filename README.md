# Application-Samples

Code samples for XIMEA cameras. Every sample is self-contained and can be used independently.

---

## Quick start

1. Install the latest [XIMEA Software Package](https://www.ximea.com/software-downloads).
2. Connect a camera.
3. Pick a sample from [Available samples](#available-samples) and follow its README.

### Interactive sample browser

To find, build and run samples from one terminal window, run this from the repository root (Python 3.11 or newer, no packages needed):

```bash
python samples_cli.py
```

Type to filter the list, move with the arrow keys and press Enter to **Build**, **Run** or **Open in explorer**. Build output goes to a `build/` folder inside each sample, with the full build log in `build/build.log`. **Run** builds the sample first when it has changed since the last build. Dependencies are downloaded only when `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` is set (see [Third-party libraries](#third-party-libraries)).

---

## Requirements

| Language | You need |
|----------|----------|
| C / C++ | CMake 3.16 or newer and a C/C++ compiler |
| C# / VB.NET | .NET SDK 10.0 or newer |
| Python | Python 3.11 or newer |

Some samples also need OpenCV, CUDA, OpenCL or an NVIDIA Jetson board; their READMEs list what they need.

---

## Third-party libraries

Some samples use an external library (libtiff, LibTiff.NET, OpenCV, numpy, ...). Nothing is downloaded from the internet unless you ask for it. By default a sample uses a copy you installed yourself; if there is none, its build (or the Python sample itself) stops and explains both ways to provide it.

To let every sample download what it needs, set the environment variable `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` to `1`. Each download is still accepted only if it matches the hash pinned in `dependencies/` (for OpenCV on Windows, in `dependencies/install-opencv-windows.ps1`); Python samples then run `pip install --require-hashes -r requirements.txt` themselves.

Everything dependency-related lives in [`dependencies/`](dependencies/README.md): the pinned versions, the NuGet settings, the files builds download (`dependencies/cache/`) and the copies you provide yourself (`dependencies/local/`: `cmake/<library>/` for a C/C++ library you built, `nuget/` for `.nupkg` files, `wheels/` for Python wheels).

```bash
export AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1   # Linux/macOS
```

```powershell
$env:AUTOMATICALLY_DOWNLOAD_DEPENDENCIES = '1'   # Windows PowerShell
```

---

## Available samples

<!-- samples-list:start -->
| Sample | API | Language | What it shows |
|--------|-----|----------|---------------|
| **capture-10-images** | xiAPI | [C](samples/xiapi/cross-platform/capture-10-images/) · [C++](samples/xiapiplus/cross-platform/capture-10-images/) · [C#](samples/xiapi-net-csharp/capture-10-images/) · [Python](samples/xiapi-python/cross-platform/capture-10-images/) | Captures 10 images. |
| **capture-10-images-to-dng** | xiAPI | [C](samples/xiapi/cross-platform/capture-10-images-to-dng/) · [C++](samples/xiapiplus/cross-platform/capture-10-images-to-dng/) · [C#](samples/xiapi-net-csharp/capture-10-images-to-dng/) · [Python](samples/xiapi-python/cross-platform/capture-10-images-to-dng/) | Captures 10 raw images and saves them as DNG files using the XIMEA DNG Store library. |
| **capture-10-images-to-file** | xiAPI | [C](samples/xiapi/cross-platform/capture-10-images-to-file/) · [C++](samples/xiapiplus/cross-platform/capture-10-images-to-file/) · [C#](samples/xiapi-net-csharp/capture-10-images-to-file/) · [Python](samples/xiapi-python/cross-platform/capture-10-images-to-file/) | Captures 10 images and saves them to file. |
| **capture-10-images-to-ram** | xiAPI | [C](samples/xiapi/cross-platform/capture-10-images-to-ram/) · [C++](samples/xiapiplus/cross-platform/capture-10-images-to-ram/) · [C#](samples/xiapi-net-csharp/capture-10-images-to-ram/) · [Python](samples/xiapi-python/cross-platform/capture-10-images-to-ram/) | Captures 10 images into RAM. |
| **error-handling** | xiAPI | [C](samples/xiapi/cross-platform/error-handling/) · [C++](samples/xiapiplus/cross-platform/error-handling/) · [C#](samples/xiapi-net-csharp/error-handling/) · [Python](samples/xiapi-python/cross-platform/error-handling/) | Shows how to detect and recover from an xiAPI error such as XI_TIMEOUT. |
| **offline-processing** | xiAPI | [C](samples/xiapi/cross-platform/offline-processing/) · [C++](samples/xiapiplus/cross-platform/offline-processing/) · [C#](samples/xiapi-net-csharp/offline-processing/) | Captures 10 RAW8 images, converts them to RGB24 with xiAPI offline processing after acquisition, and saves them as TIFF files. |
| **capture-25-images-in-parallel** | xiAPIplus | [C++](samples/xiapiplus/cross-platform/capture-25-images-in-parallel/) | Opens all connected cameras in parallel, triggers a master camera by software and the slaves through its GPO1 exposure pulse, and captures 5 bursts of 5 frames per camera on per-camera acquisition threads with a shared worker pool. |
| **capture-50-images-and-display-opencv** | xiAPIplus | [C++](samples/xiapiplus/cross-platform/capture-50-images-and-display-opencv/) | Captures 50 images and displays them using OpenCV. |
| **capture-50-images-to-video-file-opencv** | xiAPIplus | [C++](samples/xiapiplus/cross-platform/capture-50-images-to-video-file-opencv/) | Captures 50 images and writes them to a video file using OpenCV. |
| **compute-histogram-cuda** | xiAPIplus | [C++](samples/xiapiplus/cross-platform/compute-histogram-cuda/) | Delivers 8-bit frames to CPU, CUDA managed, mapped host or GPU memory with XI_PRM_TRANSPORT_DATA_TARGET, computes a 64-bin histogram of each frame with a CUDA kernel and prints it with the processing time and throughput. |
| **demosaic-images-opencl** | xiAPIplus | [C++](samples/xiapiplus/cross-platform/demosaic-images-opencl/) | Captures RAW8 Bayer frames from a color camera, demosaics them with a Malvar-He-Cutler OpenCL kernel on the GPU, saves them as RGB TIFF files and prints the kernel and transfer times. |
| **capture-10-concatenated-images-to-file** | GenTL | [C++](samples/gentl/cross-platform/capture-10-concatenated-images-to-file/) | Captures 10 images, concatenates them and saves the result to a file. |
| **capture-10-images-python** | GenTL | [Python](samples/gentl/cross-platform/capture-10-images-python/) | Loads the XIMEA GenTL producer with the harvesters GenTL consumer, sets the exposure and pixel format through the GenICam node map, captures 10 images and prints the size, timestamp and first pixels of each one. |
| **capture-10-images-to-file** | GenTL | [C++](samples/gentl/cross-platform/capture-10-images-to-file/) | Loads the XIMEA GenTL producer (`.cti`) at run time, opens the first camera through the GenTL module hierarchy, captures 10 images into application buffers and saves them as TIFF files. |
| **capture-50-images-zero-copy** | GenTL | [C++](samples/gentl/cross-platform/capture-50-images-zero-copy/) | Captures 50 images using a GenTL-based interface. |
| **averaging-camera-client** | xiAPI + MultiRecorder | [C++](samples/multirecorder/cross-platform/averaging-camera-client/) | MultiRecorder sample/application with camera-side processing. |
| **averaging-camera-server** | xiAPI + MultiRecorder | [C++](samples/multirecorder/cross-platform/averaging-camera-server/) | MultiRecorder sample/application with camera-side processing. |
| **export-recorded-session** | MultiRecorder | [XML](samples/multirecorder/cross-platform/export-recorded-session/) | Records 10 images into a saved recording session, then exports the session to TIFF files in a second MultiRecorder run. |
| **record-10-images** | MultiRecorder | [XML](samples/multirecorder/cross-platform/record-10-images/) | Records 10 images from the first connected camera with a 200 ms exposure and fixed white balance, then exports them as RGB TIFF files. |
| **record-concatenated-images** | MultiRecorder | [XML](samples/multirecorder/cross-platform/record-concatenated-images/) | Records 10 images in buffers of 2 concatenated images each and exports them as RGB TIFF files. |
| **record-from-multiple-cameras** | MultiRecorder | [XML](samples/multirecorder/cross-platform/record-from-multiple-cameras/) | Records 10 images from each of three cameras connected to this computer, with different settings, storages and export formats per camera. |
| **record-through-local-server** | MultiRecorder | [XML](samples/multirecorder/cross-platform/record-through-local-server/) | Starts a MultiRecorder camera server on this computer, records 10 images from its camera through the server, then stops the server. |
| **record-through-remote-server** | MultiRecorder | [XML](samples/multirecorder/cross-platform/record-through-remote-server/) | Records 10 images from a camera connected to another computer, which runs a MultiRecorder camera server. |
| **record-with-camera-settings-file** | MultiRecorder | [XML](samples/multirecorder/cross-platform/record-with-camera-settings-file/) | Applies camera settings saved by xiCamTool from `camera-settings.xicamera`, overrides the exposure, records 10 images and exports them as RGB TIFF files. |
| **display-images-in-window** | xiAPI.NET | [C#](samples/xiapi-net-csharp/display-images-in-window/) | Displays captured images in a window. |
| **capture-100-images-to-file** | xiAPI.NET | [VB.NET](samples/xiapi-net-vb/capture-100-images-to-file/) | Captures 100 images and saves them to file. |
| **capture-from-two-cameras** | xiAPI Python | [Python](samples/xiapi-python/cross-platform/capture-from-two-cameras/) | This code captures images simultaneously from two Ximea cameras and prints their first 10 pixel values. |
| **display-image-opencv** | xiAPI Python | [Python](samples/xiapi-python/cross-platform/display-image-opencv/) | Captures one MONO8, RGB24 or MONO16 image and shows it in an OpenCV window scaled to fit the screen. |
| **display-image-pillow** | xiAPI Python | [Python](samples/xiapi-python/cross-platform/display-image-pillow/) | Captures one MONO8, RGB24 or RGB32 image, converts it to a Pillow image and opens it in the system image viewer. |
| **display-video-opencv** | xiAPI Python | [Python](samples/xiapi-python/cross-platform/display-video-opencv/) | Captures video from a XIMEA camera, displays it with elapsed time overlay, and handles keyboard interrupt. |
| **ffs-parameter-preset** | xiAPI Python | [Python](samples/xiapi-python/cross-platform/ffs-parameter-preset/) | Saves a name=value parameter preset into the camera Flash File System (FFS) file User1, shows it, and applies it back to the camera. |
| **save-image-pillow** | xiAPI Python | [Python](samples/xiapi-python/cross-platform/save-image-pillow/) | Captures one image (RGB24 on color cameras, MONO8 on monochrome cameras) and saves it with Pillow, which picks the file format from the extension. |
| **jetson-demosaic-sample** | xiAPI, NVIDIA Jetson | [C++](samples/xiapi/hardware-specific/jetson-demosaic-sample/) | Depacking and demosaicing image from a color camera using CUDA. Render the image to screen and save to .tif |
| **gpio-samples-jetson** | xiAPI Python, NVIDIA Jetson | [Python](samples/xiapi-python/hardware-specific/gpio-samples-jetson/) | Blinks LED on XEC-NX-3P-X2G3 carrier board |
<!-- samples-list:end -->

---

## Documentation

- [XIMEA API Manual](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA .NET API Manual](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Python API Manual](https://www.ximea.com/support/wiki/apis/XiAPI_Python_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)

## License

See [LICENSE](LICENSE).
