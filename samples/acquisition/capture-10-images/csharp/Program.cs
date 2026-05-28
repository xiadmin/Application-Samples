// Opens the first available XIMEA camera, sets exposure to 100 ms,
// captures 10 frames, prints per-frame metadata, then closes.

#nullable enable

using System;
using xiApi.NET;

const int frameCount    = 10;
const int exposureUs    = 100_000;  // 100 ms
const int grabTimeoutMs = 5_000;    // must exceed exposure

var cam = new xiCam();
bool isDeviceOpen = false;
try
{
    cam.GetNumberDevices(out int numDevices);

    if (numDevices == 0)
    {
        Console.Error.WriteLine("Error: no XIMEA cameras detected");
        return 1;
    }

    Console.WriteLine($"Found {numDevices} camera(s), opening index 0");
    cam.OpenDevice(0);
    isDeviceOpen = true;

    cam.SetParam(PRM.EXPOSURE, exposureUs);
    Console.WriteLine($"Exposure: {exposureUs} us ({exposureUs / 1000} ms)");

    cam.StartAcquisition();
    Console.WriteLine($"Capturing {frameCount} frames");

    for (int i = 0; i < frameCount; i++)
    {
        xiApi.XI_IMG img = cam.GetXI_IMG(grabTimeoutMs);
        Console.WriteLine($"Frame {i + 1}/{frameCount}: {img.width}x{img.height} nframe={img.acq_nframe}");
    }

    cam.StopAcquisition();
    Console.WriteLine("Done");
    return 0;
}
catch (xiExc ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 1;
}
finally
{
    if (isDeviceOpen)
        cam.CloseDevice();
}
