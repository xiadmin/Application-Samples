#nullable enable

using System;
using xiApi.NET;

// TODO: implement the sample.

var cam = new xiCam();
bool isDeviceOpen = false;
try
{
    cam.GetNumberDevices(out int numDevices);

    if (numDevices == 0)
    {
        Console.Error.WriteLine("Error: no XIMEA cameras detected.");
        return 1;
    }

    Console.WriteLine($"Found {numDevices} camera(s), opening index 0.");
    cam.OpenDevice(0);
    isDeviceOpen = true;

    Console.WriteLine("{{binary_name}}: not yet implemented");
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
