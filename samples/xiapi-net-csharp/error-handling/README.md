# error-handling — C# sample

Shows how to detect and recover from an xiAPI error such as XI_TIMEOUT.

| Item | Value |
|------|-------|
| Category | Error handling |
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
cd samples\xiapi-net-csharp\error-handling
dotnet build ErrorHandlingCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

NuGet packages restore into `.deps\nuget-packages\` at the repository root, shared by every C# sample, from the sources listed in `nuget.config`. Package versions come from `dependencies\Directory.Packages.props`, so a `PackageReference` in this project carries no `Version`.

---

## Run

```powershell
.\samples\xiapi-net-csharp\error-handling\.dotnet-tmp\error-handling-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\error-handling
dotnet run --project ErrorHandlingCsharp.csproj
```

---

## Expected output

The sample first calls the API without an open camera, which fails with `XI_INVALID_HANDLE` and needs no hardware. It then opens the camera with a 100 ms exposure and waits only 1 ms for the first frame. The resulting `XI_TIMEOUT` is caught and handled by retrying with a 5000 ms timeout. Any other error ends the sample with a non-zero exit code; on success it exits with 0.

```text
Caught XI_INVALID_HANDLE (1) from a call without an open camera
Opening camera index 0
Caught XI_TIMEOUT (10) with a 1 ms timeout; retrying with 5000 ms
Got frame 1280x1024 nframe=1
Done
```

Without a camera, only the first line is printed, followed by `Error: no XIMEA cameras detected` on stderr and exit code 1.

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- xiAPI.NET throws `xiExc`; its `ErrorCode` is compared with `ErrorCodesEnum` in an exception filter (`catch ... when`), so only `XI_TIMEOUT` is handled locally.
- The `XI_TIMEOUT` path depends on the first frame taking longer than 1 ms. If a frame does arrive in time, the sample prints `A frame arrived within 1 ms; no timeout this time` and continues.
- The sample always opens camera index 0.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
