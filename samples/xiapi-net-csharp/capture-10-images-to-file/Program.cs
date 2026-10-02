/*
 * Sample name: Capture-10-images-to-file
 * Category: Basic acquisition / image capture
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Captures 10 images and saves them as TIFF files using
 * LibTiff.NET.
 */

#nullable enable

using System;
using System.IO;
using BitMiracle.LibTiff.Classic;
using xiApi.NET;

const int FrameCount = 10;
const int AutoExposureFrameCount = 10;
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

    cam.SetParam(PRM.EXPOSURE, ExposureUs);
    cam.SetParam(PRM.AEAG, 1);
    // MONO8 works on every camera; a color sensor delivers its luminance.
    cam.SetParam(PRM.IMAGE_DATA_FORMAT, IMG_FORMAT.MONO8);

    cam.StartAcquisition();
    isAcquiring = true;

    // Discard the first frames while auto exposure settles.
    for (int i = 0; i < AutoExposureFrameCount; i++)
    {
        cam.GetImage(out byte[] _, GrabTimeoutMs);
    }

    for (int i = 0; i < FrameCount; i++)
    {
        // This overload copies the frame, including line padding, out of the
        // driver buffer, so it works with the default (unsafe) buffer policy.
        cam.GetImage(out byte[] pixels, GrabTimeoutMs);
        ImgParams frame = cam.GetLastImageParams();

        string fileName = $"image{i:D3}.tif";
        WriteMono8Tiff(fileName, pixels, frame.GetWidth(), frame.GetHeight(), frame.GetPaddingX());
        Console.WriteLine($"Saved frame {i + 1}/{FrameCount} to {fileName}");
    }
}
catch (xiExc ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    exitCode = 1;
}
catch (IOException ex)
{
    Console.Error.WriteLine($"Error: could not write TIFF file: {ex.Message}");
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

// Writes an uncompressed 8-bit grayscale TIFF, dropping the line padding.
static void WriteMono8Tiff(string fileName, byte[] pixels, int width, int height, int paddingX)
{
    using Tiff? tiff = Tiff.Open(fileName, "w");
    if (tiff is null)
    {
        throw new IOException($"could not open '{fileName}' for writing");
    }

    tiff.SetField(TiffTag.IMAGEWIDTH, width);
    tiff.SetField(TiffTag.IMAGELENGTH, height);
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
    tiff.SetField(TiffTag.BITSPERSAMPLE, 8);
    tiff.SetField(TiffTag.ORIENTATION, Orientation.TOPLEFT);
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE);
    tiff.SetField(TiffTag.ROWSPERSTRIP, height);

    int stride = width + paddingX;
    var row = new byte[width];
    for (int y = 0; y < height; y++)
    {
        Buffer.BlockCopy(pixels, y * stride, row, 0, width);
        if (!tiff.WriteScanline(row, y))
        {
            throw new IOException($"could not write row {y} of '{fileName}'");
        }
    }
}
