# display-images-in-window — C# sample

Shows live camera images in a Windows Forms window with an exposure slider.

| Item | Value |
|------|-------|
| Category | Image display |
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
| .NET SDK | 8.0 or newer |
| Desktop runtime | .NET 8 Desktop Runtime (Windows Forms) |
| Libraries | None |

`xiApi.NETX64.dll` comes from the XIMEA Software Package (SP) installation, referenced via `XIMEA_SP_PATH`. It is not a NuGet package.

---

## Build

```powershell
cd samples\xiapi-net-csharp\display-images-in-window
dotnet build DisplayImagesInWindowCsharp.csproj -c Release --output .dotnet-tmp
```

The binary lands in `.dotnet-tmp\` inside the sample folder.

---

## Run

```powershell
cd samples\xiapi-net-csharp\display-images-in-window
.\.dotnet-tmp\display-images-in-window-csharp.exe
```

Or use `dotnet run` (no separate build step needed):

```powershell
cd samples\xiapi-net-csharp\display-images-in-window
dotnet run --project DisplayImagesInWindowCsharp.csproj
```

---

## Expected output

A window titled **XIMEA live preview** shows live MONO8 frames, scaled to fit, with an exposure slider below them. The slider covers the camera's minimum exposure up to 100 ms and starts at 50 ms. Moving it changes the exposure from the next frame on. Closing the window stops acquisition and closes the camera.

```text
Opening camera index 0
Done
```

Without a camera, the sample prints `Error: no XIMEA cameras detected` on stderr and exits with code 1 before any window opens.

---

## Known limitations / caveats

- Windows-only: Windows Forms and the XIMEA .NET wrapper are not available for Linux or macOS.
- Frames are grabbed on a background task and handed to the UI thread with `Invoke()`, which replaces the `Application.DoEvents()` loop of the original sample. Exposure changes go to the grab task through `Interlocked`, so only one thread calls into the camera.
- The net7.0 `xiApi.NETX64.dll` has no `Bitmap` overloads; `PreviewForm.CreateBitmap()` builds an 8-bit grayscale bitmap from the frame bytes and drops the line padding.
- The project builds as a console application (`Exe`), so errors also appear on stderr next to the window.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [Windows Forms](https://learn.microsoft.com/dotnet/desktop/winforms/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
