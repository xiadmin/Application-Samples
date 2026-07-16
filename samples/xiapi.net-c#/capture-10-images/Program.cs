// Opens the first available XIMEA camera, sets exposure to 100 ms,
// captures 10 frames, prints per-frame metadata, then closes.

#nullable enable

using System;
using xiApi.NET;

const int FrameCount = 10;
const int ExposureUs = 100_000;  // 100 ms
const int GrabTimeoutMs = 5_000; // must exceed exposure

var cam = new xiCam();
bool isDeviceOpen = false;
bool isAcquiring = false;

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

    cam.SetParam(PRM.EXPOSURE, ExposureUs);
    Console.WriteLine($"Exposure: {ExposureUs} us ({ExposureUs / 1000} ms)");

    cam.StartAcquisition();
    isAcquiring = true;
    Console.WriteLine($"Capturing {FrameCount} frames");

    for (int i = 0; i < FrameCount; i++)
    {
        xiApi.XI_IMG img = cam.GetXI_IMG(GrabTimeoutMs);
        Console.WriteLine($"Frame {i + 1}/{FrameCount}: {img.width}x{img.height} nframe={img.acq_nframe}");
    }

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
    if (isAcquiring)
    {
        cam.StopAcquisition();
    }

    if (isDeviceOpen)
    {
        cam.CloseDevice();
    }
}
