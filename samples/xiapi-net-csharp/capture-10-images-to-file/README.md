# capture-10-images-to-file — C# sample

Captures 10 images and saves them as TIFF files using LibTiff.NET.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| OS platform | Windows |
| Hardware platform | Cross-platform |
| API type | xiAPI.NET |
| Language | C# |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| .NET SDK | 8.0 or newer |
| Libraries | BitMiracle.LibTiff.NET (version pinned in `dependencies/Directory.Packages.props`) |
| Network | Required for the first NuGet restore into `.deps/nuget-packages/`; `dotnet build --no-restore` works offline afterwards. |

`xiApi.NETX64.dll` comes from the XIMEA Software Package (SP) installation, referenced via `XIMEA_SP_PATH`. It is not a NuGet package.

---

## Build

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-file
dotnet build Capture10ImagesToFileCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

NuGet packages restore into `.deps\nuget-packages\` at the repository root, shared by every C# sample, from the sources listed in `nuget.config`. Package versions come from `dependencies\Directory.Packages.props`, so a `PackageReference` in this project carries no `Version`.

---

## Run

Run from the directory where the TIFF files should be created:

```powershell
.\samples\xiapi-net-csharp\capture-10-images-to-file\.dotnet-tmp\capture-10-images-to-file-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-file
dotnet run --project Capture10ImagesToFileCsharp.csproj
```

---

## Expected output

The sample enables automatic exposure/gain, discards 10 warm-up frames, and saves the next 10 frames as 8-bit grayscale `image000.tif` through `image009.tif`.

```text
Opening camera index 0
Saved frame 1/10 to image000.tif
...
Saved frame 10/10 to image009.tif
Done
```

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- All cameras deliver MONO8; a color camera saves its luminance. See the C, C++ or Python sample for an RGB24 variant.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- Frames are read with `xiCam.GetImage(out byte[], timeout)`, which copies the frame and its line padding out of the driver buffer; the TIFF writer drops the padding.
- The project targets net8.0 but links against the net7.0 `xiApi.NETX64.dll` (the latest version shipped with the SP). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [LibTiff.NET](https://bitmiracle.com/libtiff/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
