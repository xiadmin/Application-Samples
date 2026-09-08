# capture-10-images — C# sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| API type | xiAPI.NET |

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
cd samples\xiapi-net-csharp\capture-10-images
dotnet build CaptureImages.csproj -c Release --output .dotnet-tmp
```

Binary lands in `.dotnet-tmp\` inside the sample folder.

---

## Run

```powershell
.\samples\xiapi-net-csharp\capture-10-images\.dotnet-tmp\capture-10-images-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\capture-10-images
dotnet run --project CaptureImages.csproj
```

---

## Expected output

```
Found 1 camera(s), opening index 0
Exposure: 100000 us (100 ms)
Capturing 10 frames
Frame 1/10: 1280x1024 nframe=1
...
Frame 10/10: 1280x1024 nframe=10
Done
```

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- The project targets net8.0 but links against the net7.0 `xiApi.NETX64.dll` (the latest
  version shipped with the SP). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
