# capture-100-images-to-file — VB.NET sample

Captures 100 MONO8 images and saves them as JPEG files with the WPF JpegBitmapEncoder.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| OS platform | Windows |
| Hardware platform | Cross-platform |
| API type | xiAPI.NET |
| Language | Visual Basic .NET |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| .NET SDK | 8.0 or newer |
| Libraries | WPF imaging (`System.Windows.Media.Imaging`, part of the .NET Windows Desktop runtime) |

`xiApi.NETX64.dll` comes from the XIMEA Software Package (SP) installation, referenced via `XIMEA_SP_PATH`. It is not a NuGet package.

---

## Build

```powershell
cd samples\xiapi-net-vb\capture-100-images-to-file
dotnet build Capture100ImagesToFileVb.vbproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

---

## Run

Run from the directory where the `images` folder should be created:

```powershell
.\samples\xiapi-net-vb\capture-100-images-to-file\.dotnet-tmp\capture-100-images-to-file-vb.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-vb\capture-100-images-to-file
dotnet run --project Capture100ImagesToFileVb.vbproj
```

---

## Expected output

The sample prints the camera name and serial number, sets a 2 ms exposure and 5 dB gain, and saves 100 8-bit grayscale frames as `images\image_000.jpg` through `images\image_099.jpg` at JPEG quality 100.

```text
Opening camera index 0
Camera MQ013MG-ON, serial number 12345678
Saved frame 1/100 (1280x1024) to images\image_000.jpg
...
Saved frame 100/100 (1280x1024) to images\image_099.jpg
Done
```

The camera name, serial number and resolution depend on the camera.

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper and WPF are not available for Linux or macOS.
- All cameras deliver MONO8; a color camera saves its luminance.
- The `images` folder is created when missing. Existing files in it with the same names are overwritten.
- The exposure and gain are fixed. Adjust `ExposureUs` and `GainDb` in `Program.vb` for brighter or darker scenes.
- Frames are read with `xiCam.GetImageByteArray` and the SAFE buffer policy into one preallocated array. `xiCam.GetImage` cannot be called from VB, because its overloads differ only in `ByRef`.
- The project targets net8.0 but links against the net7.0 `xiApi.NETX64.dll` (the latest version shipped with the SP). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [JpegBitmapEncoder](https://learn.microsoft.com/dotnet/api/system.windows.media.imaging.jpegbitmapencoder)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
