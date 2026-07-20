// Sample name: Capture-10-images
// Category: Basic acquisition / image capture
// OS platform: Cross-Platform
// Hardware platform: Cross-platform
// API type: xiAPI.NET
// Short description: Captures 10 images.
//
// Copyright (c) 2026 XIMEA s.r.o.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

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
