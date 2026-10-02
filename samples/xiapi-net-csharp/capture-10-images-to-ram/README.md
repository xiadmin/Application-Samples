# capture-10-images-to-ram — C# sample

Captures 10 images into application-owned RAM.

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

`xiApi.NETX64.dll` comes from the XIMEA Software Package (SP) installation, referenced via `XIMEA_SP_PATH`. It is not a NuGet package.

---

## Build

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-ram
dotnet build Capture10ImagesToRamCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

NuGet packages restore into `.deps\nuget-packages\` at the repository root, shared by every C# sample, from the sources listed in `nuget.config`. Package versions come from `dependencies\Directory.Packages.props`, so a `PackageReference` in this project carries no `Version`.

---

## Run

```powershell
.\samples\xiapi-net-csharp\capture-10-images-to-ram\.dotnet-tmp\capture-10-images-to-ram-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\capture-10-images-to-ram
dotnet run --project Capture10ImagesToRamCsharp.csproj
```

---

## Expected output

The sample captures 10 RAW8 frames into managed `byte[]` arrays, closes the camera, and then prints per-frame metadata from RAM.

```text
Opening camera index 0
Camera closed; reading retained images from RAM
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Frames are read with `xiCam.GetImage(out byte[], timeout)`, which allocates a new array per frame and copies the data, including line padding.
- The project targets net8.0 but links against the net7.0 `xiApi.NETX64.dll` (the latest version shipped with the SP). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
