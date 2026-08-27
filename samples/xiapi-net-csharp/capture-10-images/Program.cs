/*
 * Sample name: Capture-10-images
 * Category: Basic acquisition / image capture
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Captures 10 images.
 */

#nullable enable

using System;
using xiApi.NET;

const int FrameCount = 10;
const int ExposureUs = 100_000;  // 100 ms
const int GrabTimeoutMs = 5_000; // must exceed exposure

var cam = new xiCam();
bool isDeviceOpen = false;
bool isAcquiring = false;
int exitCode = 0;

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
}
catch (xiExc ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    exitCode = 1;
}
finally
{
    if (isAcquiring)
    {
        try
        {
            cam.StopAcquisition();
        }
        catch (xiExc ex)
        {
            Console.Error.WriteLine($"Error while stopping acquisition: {ex.Message}");
            exitCode = 1;
        }
    }

    if (isDeviceOpen)
    {
        try
        {
            cam.CloseDevice();
        }
        catch (xiExc ex)
        {
            Console.Error.WriteLine($"Error while closing device: {ex.Message}");
            exitCode = 1;
        }
    }
}

if (exitCode == 0)
{
    Console.WriteLine("Done");
}

return exitCode;
