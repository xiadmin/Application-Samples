/*
 * Sample name: Offline-processing
 * Category: Image processing
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Captures 10 RAW8 images, converts them to RGB24 with
 * xiAPI offline processing after acquisition, and saves them as TIFF files.
 */

#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using BitMiracle.LibTiff.Classic;
using xiApi.NET;

const int FrameCount = 10;
const int ExposureUs = 10_000;
const int GrabTimeoutMs = 5_000;
const int ProcessTimeoutMs = 5_000;

var cam = new xiCam();
bool isDeviceOpen = false;
bool isAcquiring = false;
var frames = new List<byte[]>(FrameCount);
int width = 0;
int height = 0;
bool isColor = false;
string context = string.Empty;
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
    isColor = cam.GetParamInt(PRM.IMAGE_IS_COLOR) == 1;

    if (isColor)
    {
        // Not applied to the RAW8 frames; recorded in the context list below
        // and applied by the offline processing.
        cam.SetParam(PRM.WB_KR, 1.05f);
        cam.SetParam(PRM.WB_KG, 1.0f);
        cam.SetParam(PRM.WB_KB, 2.10f);
    }

    cam.StartAcquisition();
    isAcquiring = true;

    for (int i = 0; i < FrameCount; i++)
    {
        cam.GetImage(out byte[] pixels, GrabTimeoutMs);
        frames.Add(pixels);
    }

    ImgParams lastFrame = cam.GetLastImageParams();
    width = lastFrame.GetWidth();
    height = lastFrame.GetHeight();

    cam.StopAcquisition();
    isAcquiring = false;

    // The context list holds every setting the frames were captured with. It
    // is all the processing needs, so the camera is closed before it runs.
    context = cam.GetParamString(PRM.API_CONTEXT_LIST);
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

Console.WriteLine($"Camera closed; processing {FrameCount} stored frames offline");

var proc = new xiProc();
bool isProcOpen = false;

try
{
    proc.ProcOpen();
    isProcOpen = true;
    proc.ProcSetParam(PRM.API_CONTEXT_LIST, context);
    proc.ProcSetParamInt(PRM.IMAGE_DATA_FORMAT, isColor ? IMG_FORMAT.RGB24 : IMG_FORMAT.MONO8);

    for (int i = 0; i < frames.Count; i++)
    {
        byte[] processed = ProcessFrame(proc, frames[i], ProcessTimeoutMs);
        string fileName = $"image{i:D3}.tif";
        WriteTiff(fileName, processed, width, height, isColor ? 3 : 1);
        Console.WriteLine($"Processed frame {i + 1}/{FrameCount} ({width}x{height}) to {fileName}");
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
    if (isProcOpen)
    {
        try
        {
            proc.ProcClose();
        }
        catch (xiExc ex)
        {
            Console.Error.WriteLine($"Error while closing processing: {ex.Message}");
            exitCode = 1;
        }
    }
}

if (exitCode == 0)
{
    Console.WriteLine("Done");
}

return exitCode;

// ProcPushImage() reads only XI_IMG.bp, so point it at the stored frame. The
// array stays pinned until the processed image has been pulled.
static unsafe byte[] ProcessFrame(xiProc proc, byte[] pixels, int timeoutMs)
{
    fixed (byte* firstPixel = pixels)
    {
        var raw = new xiApi.XI_IMG();
        raw.Clear();
        raw.bp = firstPixel;
        raw.bp_size = pixels.Length;
        proc.ProcPushImage(raw);
        proc.ProcPullImage(out byte[] processed, timeoutMs);
        return processed;
    }
}

// Writes an uncompressed 8-bit TIFF with one (gray) or three (RGB) channels.
// xiAPI delivers RGB24 as BGR with optional line padding; TIFF expects RGB.
static void WriteTiff(string fileName, byte[] pixels, int width, int height, int channelCount)
{
    using Tiff? tiff = Tiff.Open(fileName, "w");
    if (tiff is null)
    {
        throw new IOException($"could not open '{fileName}' for writing");
    }

    tiff.SetField(TiffTag.IMAGEWIDTH, width);
    tiff.SetField(TiffTag.IMAGELENGTH, height);
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, channelCount);
    tiff.SetField(TiffTag.BITSPERSAMPLE, 8);
    tiff.SetField(TiffTag.ORIENTATION, Orientation.TOPLEFT);
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
    tiff.SetField(TiffTag.PHOTOMETRIC, channelCount == 3 ? Photometric.RGB : Photometric.MINISBLACK);
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE);
    tiff.SetField(TiffTag.ROWSPERSTRIP, height);

    int rowByteCount = width * channelCount;
    int stride = pixels.Length / height;
    var row = new byte[rowByteCount];
    for (int y = 0; y < height; y++)
    {
        int rowStart = y * stride;
        if (channelCount == 3)
        {
            for (int x = 0; x < rowByteCount; x += 3)
            {
                row[x] = pixels[rowStart + x + 2];
                row[x + 1] = pixels[rowStart + x + 1];
                row[x + 2] = pixels[rowStart + x];
            }
        }
        else
        {
            Buffer.BlockCopy(pixels, rowStart, row, 0, rowByteCount);
        }

        if (!tiff.WriteScanline(row, y))
        {
            throw new IOException($"could not write row {y} of '{fileName}'");
        }
    }
}
