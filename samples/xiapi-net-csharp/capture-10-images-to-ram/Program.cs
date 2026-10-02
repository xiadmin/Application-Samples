/*
 * Sample name: Capture-10-images-to-ram
 * Category: Basic acquisition / image capture
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Captures 10 images into application-owned RAM.
 */

#nullable enable

using System;
using System.Collections.Generic;
using Ximea.Samples;
using xiApi.NET;

const int FrameCount = 10;
const int ExposureUs = 10_000;
const int GrabTimeoutMs = 5_000;

var cam = new xiCam();
bool isDeviceOpen = false;
bool isAcquiring = false;
var frames = new List<StoredFrame>(FrameCount);
int exitCode = 0;

try
{
    cam.GetNumberDevices(out int numDevices);

    if (numDevices == 0)
    {
        Console.Error.WriteLine("Error: no XIMEA cameras detected");
        return 1;
    }

    Console.WriteLine("Opening camera index 0");
    cam.OpenDevice(0);
    isDeviceOpen = true;

    cam.SetParam(PRM.IMAGE_DATA_FORMAT, IMG_FORMAT.RAW8);
    cam.SetParam(PRM.EXPOSURE, ExposureUs);

    cam.StartAcquisition();
    isAcquiring = true;

    for (int i = 0; i < FrameCount; i++)
    {
        // The out overload allocates a managed array and copies the frame into
        // it, so the data stays valid after the driver reuses its buffer.
        cam.GetImage(out byte[] pixels, GrabTimeoutMs);
        ImgParams frame = cam.GetLastImageParams();
        frames.Add(new StoredFrame(frame.GetWidth(), frame.GetHeight(), frame.GetFrameNum(), pixels));
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

if (exitCode != 0)
{
    return exitCode;
}

Console.WriteLine("Camera closed; reading retained images from RAM");
for (int i = 0; i < frames.Count; i++)
{
    StoredFrame frame = frames[i];
    int firstByte = frame.Pixels.Length > 0 ? frame.Pixels[0] : -1;
    Console.WriteLine($"Frame {i + 1}/{FrameCount}: {frame.Width}x{frame.Height} nframe={frame.FrameNumber} first_byte={firstByte}");
}

Console.WriteLine("Done");
return 0;
