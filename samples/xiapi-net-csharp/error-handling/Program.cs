/*
 * Sample name: Error-handling
 * Category: Error handling
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Shows how to detect and recover from an xiAPI error such
 * as XI_TIMEOUT.
 */

#nullable enable

using System;
using xiApi.NET;

const int ExposureUs = 100_000;
const int ShortTimeoutMs = 1; // far shorter than the exposure, so the first frame cannot arrive in time
const int GrabTimeoutMs = 5_000;

const int XiInvalidHandle = (int)ErrorCodesEnum.XI_INVALID_HANDLE;
const int XiTimeout = (int)ErrorCodesEnum.XI_TIMEOUT;

// 1. Every failed xiAPI.NET call throws xiExc with the XI_RETURN code in
//    ErrorCode. A camera object that was never opened fails with
//    XI_INVALID_HANDLE; no camera is needed.
try
{
    var unopened = new xiCam();
    unopened.GetParamInt(PRM.EXPOSURE);
    Console.Error.WriteLine("Error: a call without an open camera succeeded unexpectedly");
    return 1;
}
catch (xiExc ex) when (ex.ErrorCode == XiInvalidHandle)
{
    Console.WriteLine($"Caught XI_INVALID_HANDLE ({ex.ErrorCode}) from a call without an open camera");
}
catch (xiExc ex)
{
    Console.Error.WriteLine($"Error: a call without an open camera failed with {ex.ErrorCode} instead of XI_INVALID_HANDLE: {ex.Message}");
    return 1;
}

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

    cam.SetParam(PRM.EXPOSURE, ExposureUs);
    cam.StartAcquisition();
    isAcquiring = true;

    // 2. Some errors are recoverable. The exception filter catches only
    //    XI_TIMEOUT; any other error reaches the outer handler.
    xiApi.XI_IMG img;
    try
    {
        img = cam.GetXI_IMG(ShortTimeoutMs);
        Console.WriteLine($"A frame arrived within {ShortTimeoutMs} ms; no timeout this time");
    }
    catch (xiExc ex) when (ex.ErrorCode == XiTimeout)
    {
        Console.WriteLine($"Caught XI_TIMEOUT ({ex.ErrorCode}) with a {ShortTimeoutMs} ms timeout; retrying with {GrabTimeoutMs} ms");
        img = cam.GetXI_IMG(GrabTimeoutMs);
    }

    Console.WriteLine($"Got frame {img.width}x{img.height} nframe={img.acq_nframe}");
}
catch (xiExc ex)
{
    Console.Error.WriteLine($"Error {ex.ErrorCode}: {ex.Message}");
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
