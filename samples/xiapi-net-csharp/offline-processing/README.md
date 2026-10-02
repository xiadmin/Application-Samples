# offline-processing — C# sample

Captures 10 RAW8 images, converts them to RGB24 with xiAPI offline processing after acquisition, and saves them as TIFF files.

| Item | Value |
|------|-------|
| Category | Image processing |
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
cd samples\xiapi-net-csharp\offline-processing
dotnet build OfflineProcessingCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

NuGet packages restore into `.deps\nuget-packages\` at the repository root, shared by every C# sample, from the sources listed in `nuget.config`. Package versions come from `dependencies\Directory.Packages.props`, so a `PackageReference` in this project carries no `Version`.

---

## Run

Run from the directory where the TIFF files should be created:

```powershell
.\samples\xiapi-net-csharp\offline-processing\.dotnet-tmp\offline-processing-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\offline-processing
dotnet run --project OfflineProcessingCsharp.csproj
```

---

## Expected output

The sample captures 10 RAW8 frames into RAM. It reads the camera's API context list (`XI_PRM_API_CONTEXT_LIST`, every setting the frames were captured with) and then closes the camera. The xiAPI offline processing (`xiProc`) then converts each stored frame using that context and saves it as `image000.tif` through `image009.tif`. Color cameras produce RGB24 with the white balance coefficients recorded in the context; monochrome cameras produce MONO8.

```text
Opening camera index 0
Camera closed; processing 10 stored frames offline
Processed frame 1/10 (1280x1024) to image000.tif
...
Processed frame 10/10 (1280x1024) to image009.tif
Done
```

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- The project enables `AllowUnsafeBlocks` for one reason: `xiProc.ProcPushImage()` reads only `XI_IMG.bp`, so `ProcessFrame()` points it at the stored array, which stays pinned until the processed image is pulled.
- The net7.0 `xiApi.NETX64.dll` pulls processed images only as `byte[]`; the sample writes TIFF through LibTiff.NET rather than `System.Drawing`.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- The white balance coefficients (1.05 / 1.0 / 2.10) are example values from the original sample; adjust them for your light source.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [LibTiff.NET](https://bitmiracle.com/libtiff/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
