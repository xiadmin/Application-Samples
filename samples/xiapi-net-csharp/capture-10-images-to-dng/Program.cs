/*
 * Sample name: Capture-10-images-to-dng
 * Category: Basic acquisition / image capture
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Captures 10 raw images and saves them as DNG files using
 * the XIMEA DNG Store library.
 */

#nullable enable

using System;
using Ximea.Samples;
using xiApi.NET;

const int FrameCount = 10;
const int ExposureUs = 10_000;
const int GrabTimeoutMs = 5_000;

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

    Console.WriteLine("Opening camera index 0");
    cam.OpenDevice(0);
    isDeviceOpen = true;

    // DNG stores the sensor data as captured: Bayer data on a color camera.
    cam.SetParam(PRM.IMAGE_DATA_FORMAT, IMG_FORMAT.RAW8);
    cam.SetParam(PRM.EXPOSURE, ExposureUs);

    cam.StartAcquisition();
    isAcquiring = true;

    // Read once; the settings do not change between frames here.
    DngStore.Metadata metadata = DngStore.FillMetadataFromCamera(cam);

    for (int i = 0; i < FrameCount; i++)
    {
        xiApi.XI_IMG image = cam.GetXI_IMG(GrabTimeoutMs);
        string fileName = $"image{i:D3}.dng";
        DngStore.StoreImage(fileName, image, metadata);
        Console.WriteLine($"Saved frame {i + 1}/{FrameCount} ({image.width}x{image.height}) to {fileName}");
    }
}
catch (xiExc ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    exitCode = 1;
}
catch (DllNotFoundException ex)
{
    Console.Error.WriteLine($"Error: could not load the XIMEA DNG Store library: {ex.Message}");
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
