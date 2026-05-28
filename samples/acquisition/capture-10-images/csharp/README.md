# capture-10-images — C# sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.32 or newer |
| .NET SDK | 8.0 or newer |


---

## Build

### Using build.ps1 (builds all samples)

```powershell
cd <repo-root>
.\build.ps1
```

Binary and supporting files land in:

```
build\acquisition-capture-10-images-csharp\
```

### Directly with dotnet

```powershell
cd samples\acquisition\capture-10-images\csharp
dotnet build CaptureImages.csproj -c Release --output .dotnet-tmp
```

Binary lands in `.dotnet-tmp\` inside the sample folder.

---

## Run

### After build.ps1

```powershell
.\build\acquisition-capture-10-images-csharp\acquisition-capture-10-images-csharp.exe
```

### After a direct dotnet build

```powershell
.\\samples\\acquisition\\capture-10-images\\csharp\\.dotnet-tmp\\acquisition-capture-10-images-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\acquisition\capture-10-images\csharp
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
  version shipped with the SDK). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
