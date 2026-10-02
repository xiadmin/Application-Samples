/*
 * Sample name: Display-images-in-window
 * Category: Image display
 * OS platform: Windows
 * Hardware platform: Cross-platform
 * API type: xiAPI.NET
 * Short description: Shows live camera images in a Windows Forms window with
 * an exposure slider.
 */

#nullable enable

using System;
using System.Windows.Forms;
using xiApi.NET;

namespace Ximea.Samples
{
    internal static class Program
    {
        // Windows Forms needs a single-threaded apartment, which top-level
        // statements cannot request, hence the explicit Main.
        [STAThread]
        private static int Main()
        {
            var cam = new xiCam();

            // Open the camera before any window exists, so a missing camera is
            // reported on stderr instead of in a dialog nobody may be watching.
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
            }
            catch (xiExc ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }

            int exitCode;
            try
            {
                Application.SetHighDpiMode(HighDpiMode.SystemAware);
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);

                using var form = new PreviewForm(cam);
                Application.Run(form);
                exitCode = form.ExitCode;
            }
            finally
            {
                try
                {
                    cam.CloseDevice();
                }
                catch (xiExc ex)
                {
                    Console.Error.WriteLine($"Error while closing device: {ex.Message}");
                }
            }

            if (exitCode == 0)
            {
                Console.WriteLine("Done");
            }

            return exitCode;
        }
    }
}
