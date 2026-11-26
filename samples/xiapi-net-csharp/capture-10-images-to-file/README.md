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
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Libraries | BitMiracle.LibTiff.NET (version pinned in `dependencies/Directory.Packages.props`) |
| .NET SDK | 10.0 or newer |
| Network | With `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` set, required for the first NuGet restore into `dependencies/cache/nuget-packages/`; `dotnet build --no-restore` works offline afterwards. Not set, no network is used. |

`xiApi.NETX64.dll` comes from the XIMEA Software Package (SP) installation, referenced via `XIMEA_SP_PATH`. It is not a NuGet package.

---

## Third-party libraries

This sample uses the NuGet package(s) **BitMiracle.LibTiff.NET**. Nothing is downloaded unless you ask for it; a build that cannot find them stops and explains both ways.

| `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` | What happens |
|---|---|
| not set (default) | NuGet restores only from `dependencies\local\nuget\` at the repository root, where you put the `.nupkg` files yourself, at any version; nothing is downloaded. Your packages are used as they are, without a signature or lock-file check. |
| `1` | NuGet restores the packages from nuget.org into `dependencies\cache\nuget-packages\`, shared by every C# sample, at the versions in `dependencies\Directory.Packages.props`. A package is used only if it carries nuget.org's repository signature (see `dependencies\nuget.config`) and matches the content hash in this sample's `packages.lock.json`. |

To let this and every other sample download what it needs, set `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` to `1` (`ON`, `TRUE` and `YES` work too) before you build or run; the commands below then work unchanged. Unset it to go back to your own copies.

```bash
export AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1   # Linux/macOS
```

```powershell
$env:AUTOMATICALLY_DOWNLOAD_DEPENDENCIES = '1'   # Windows PowerShell
```

---

## Build

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-file
dotnet build Capture10ImagesToFileCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

NuGet packages restore into `dependencies\cache\nuget-packages\` at the repository root, shared by every C# sample: from nuget.org when `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` is set, otherwise only from `dependencies\local\nuget\`, where you put the `.nupkg` files yourself; see [Third-party libraries](#third-party-libraries). Package versions come from `dependencies\Directory.Packages.props`, so a `PackageReference` in this project carries no `Version`.

---

## Run

Output files are written to the current directory (the sample folder in the commands below).

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-file
.\.dotnet-tmp\capture-10-images-to-file-csharp.exe
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
- The project targets net10.0 but links against the net7.0 `xiApi.NETX64.dll` (the latest version shipped with the SP). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [LibTiff.NET](https://bitmiracle.com/libtiff/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
