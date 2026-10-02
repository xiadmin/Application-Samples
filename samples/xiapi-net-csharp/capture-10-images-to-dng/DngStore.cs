#nullable enable

using System;
using System.Runtime.InteropServices;
using xiApi;
using xiApi.NET;

namespace Ximea.Samples
{
    /// <summary>
    /// P/Invoke wrapper for xiapi_dng_store.dll from the "DNG Store" component of
    /// the XIMEA Software Package. The project copies the DLL next to the sample.
    /// </summary>
    internal static class DngStore
    {
        private const string LibraryName = "xiapi_dng_store";
        private const int XiOk = 0;

        /// <summary>Mirror of XI_DNG_METADATA in xiapi_dng_store.h.</summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        internal struct Metadata
        {
            public int Cfa; // XI_COLOR_FILTER_ARRAY
            public int MaxPixelValue;

            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string CameraModelName;

            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string CameraSerialNumber;

            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string CameraUserId;

            public int AutoExposureOn; // 1: on, 0: off, -1: N/A
            public int AutoWhiteBalanceOn;
            public float WhiteBalanceR;
            public float WhiteBalanceG;
            public float WhiteBalanceB;
            public float LensAperture;
            public float LensFocalLength;
            public int AcqTimeYear;
            public int AcqTimeMonth;
            public int AcqTimeDay;
            public int AcqTimeHour;
            public int AcqTimeMinute;
            public int AcqTimeSecond;

            // The library writes these two as well; without them it would
            // write past the end of the managed structure.
            public IntPtr AdditionalPrivateData;
            public uint AdditionalPrivateDataSize;
        }

        /// <summary>Reads model, serial number, CFA, white balance and the like from an open camera.</summary>
        internal static Metadata FillMetadataFromCamera(xiCam cam)
        {
            var metadata = new Metadata();
            Check(NativeMethods.xidngFillMetadataFromCameraParams(cam.GetApiHandle(), ref metadata),
                  "xidngFillMetadataFromCameraParams");
            return metadata;
        }

        /// <summary>Writes an image captured with xiCam.GetXI_IMG() as a DNG file.</summary>
        internal static void StoreImage(string fileName, XI_IMG image, Metadata metadata)
        {
            Check(NativeMethods.xidngStore(fileName, in image, in metadata), "xidngStore");
        }

        // The library returns XI_RETURN codes; report them like xiAPI.NET does.
        private static void Check(int result, string location)
        {
            if (result != XiOk)
            {
                throw new xiExc(result, location);
            }
        }

        private static class NativeMethods
        {
            // The C API takes pointers, so the structures are passed by reference.
            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
            internal static extern int xidngFillMetadataFromCameraParams(UIntPtr cameraHandle, ref Metadata metadata);

            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
            internal static extern int xidngStore(
                [MarshalAs(UnmanagedType.LPUTF8Str)] string fileName, in XI_IMG image, in Metadata metadata);
        }
    }
}
