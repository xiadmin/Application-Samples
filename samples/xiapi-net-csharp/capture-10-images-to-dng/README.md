# capture-10-images-to-dng — C# sample

Captures 10 raw images and saves them as DNG files using the XIMEA DNG Store library.

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
| DNG Store | The "DNG Store" component of the XIMEA SP (installed by default) |

`xiApi.NETX64.dll` and `xiapi_dng_store.dll` come from the XIMEA Software Package (SP) installation, referenced via `XIMEA_SP_PATH`. The build copies `xiapi_dng_store.dll` next to the sample and fails if it is missing.

---

## Build

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-dng
dotnet build Capture10ImagesToDngCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

NuGet packages restore into `.deps\nuget-packages\` at the repository root, shared by every C# sample, from the sources listed in `nuget.config`. Package versions come from `dependencies\Directory.Packages.props`, so a `PackageReference` in this project carries no `Version`.

---

## Run

Run from the directory where the DNG files should be created:

```powershell
.\samples\xiapi-net-csharp\capture-10-images-to-dng\.dotnet-tmp\capture-10-images-to-dng-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-dng
dotnet run --project Capture10ImagesToDngCsharp.csproj
```

---

## Expected output

The sample captures 10 RAW8 frames, Bayer data on a color camera, and saves each as `image000.dng` through `image009.dng`. Camera metadata (model, serial number, color filter array, white balance) is read once with `xidngFillMetadataFromCameraParams` and written into every file. DNG files open in raw converters such as Adobe Lightroom, darktable or RawTherapee.

```text
Opening camera index 0
Saved frame 1/10 (1280x1024) to image000.dng
...
Saved frame 10/10 (1280x1024) to image009.dng
Done
```

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- `DngStore.cs` declares the P/Invoke calls. It passes `XI_IMG` and the metadata by reference, as the C API expects, and includes the trailing `additionalPrivateData` fields that the library writes.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Existing `image000.dng` through `image009.dng` files in the current directory are overwritten.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
