#nullable enable

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using xiApi.NET;

namespace Ximea.Samples
{
    /// <summary>
    /// Shows live MONO8 frames from an open camera, with an exposure slider.
    /// Frames are grabbed on a background task and handed to the UI thread.
    /// </summary>
    internal sealed class PreviewForm : Form
    {
        private const int InitialExposureUs = 50_000;
        private const int MaxSliderExposureUs = 100_000; // keeps the preview responsive
        private const int GrabTimeoutMs = 5_000;
        private const int NoPendingExposure = -1;

        private readonly xiCam _cam;
        private readonly PictureBox _pictureBox = new PictureBox();
        private readonly TrackBar _exposureBar = new TrackBar();
        private readonly Label _exposureLabel = new Label();
        private readonly CancellationTokenSource _cancellation = new CancellationTokenSource();
        private Task? _grabLoop;
        private bool _isAcquiring;

        // Written by the UI thread, applied by the grab loop between frames, so
        // only one thread ever calls into the camera.
        private int _pendingExposureUs = NoPendingExposure;

        internal PreviewForm(xiCam cam)
        {
            _cam = cam;

            Text = "XIMEA live preview";
            ClientSize = new Size(800, 640);

            _pictureBox.Dock = DockStyle.Fill;
            _pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            _pictureBox.BackColor = Color.Black;

            _exposureLabel.Dock = DockStyle.Bottom;
            _exposureLabel.TextAlign = ContentAlignment.MiddleCenter;
            _exposureLabel.Height = 24;

            _exposureBar.Dock = DockStyle.Bottom;
            _exposureBar.TickStyle = TickStyle.None;
            _exposureBar.Scroll += OnExposureScroll;

            Controls.Add(_pictureBox);
            Controls.Add(_exposureLabel);
            Controls.Add(_exposureBar);
        }

        /// <summary>0 when the preview closed normally, 1 after a camera error.</summary>
        internal int ExitCode { get; private set; }

        protected override async void OnShown(EventArgs e)
        {
            base.OnShown(e);

            try
            {
                _cam.SetParam(PRM.IMAGE_DATA_FORMAT, IMG_FORMAT.MONO8);
                _cam.SetParam(PRM.EXPOSURE, InitialExposureUs);

                int minExposureUs = _cam.GetParamInt(PRM.EXPOSURE_MIN);
                int maxExposureUs = Math.Min(_cam.GetParamInt(PRM.EXPOSURE_MAX), MaxSliderExposureUs);
                _exposureBar.Minimum = minExposureUs;
                _exposureBar.Maximum = maxExposureUs;
                _exposureBar.SmallChange = Math.Max(1, (maxExposureUs - minExposureUs) / 100);
                _exposureBar.LargeChange = Math.Max(1, (maxExposureUs - minExposureUs) / 10);
                _exposureBar.Value = Math.Clamp(InitialExposureUs, minExposureUs, maxExposureUs);
                ShowExposure(_exposureBar.Value);

                _cam.StartAcquisition();
                _isAcquiring = true;

                CancellationToken token = _cancellation.Token;
                _grabLoop = Task.Run(() => GrabLoop(token));
                await _grabLoop;
            }
            catch (xiExc ex)
            {
                ExitCode = 1;
                Console.Error.WriteLine($"Error: {ex.Message}");
                MessageBox.Show(this, ex.Message, "Camera error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
        }

        protected override async void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);

            if (_grabLoop is not null && !_grabLoop.IsCompleted)
            {
                // Keep the form alive until the grab loop has delivered its last
                // frame; Invoke() on a disposed form would throw.
                e.Cancel = true;
                _cancellation.Cancel();
                await Task.WhenAny(_grabLoop);
                Close();
                return;
            }

            if (_isAcquiring)
            {
                _isAcquiring = false;
                try
                {
                    _cam.StopAcquisition();
                }
                catch (xiExc ex)
                {
                    ExitCode = 1;
                    Console.Error.WriteLine($"Error while stopping acquisition: {ex.Message}");
                }
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _pictureBox.Image?.Dispose();
                _cancellation.Dispose();
            }

            base.Dispose(disposing);
        }

        // Runs on a thread-pool thread. Exceptions surface through the awaited
        // task in OnShown.
        private void GrabLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                int exposureUs = Interlocked.Exchange(ref _pendingExposureUs, NoPendingExposure);
                if (exposureUs != NoPendingExposure)
                {
                    _cam.SetParam(PRM.EXPOSURE, exposureUs);
                }

                _cam.GetImage(out byte[] pixels, GrabTimeoutMs);
                ImgParams frame = _cam.GetLastImageParams();
                Bitmap bitmap = CreateBitmap(pixels, frame.GetWidth(), frame.GetHeight(), frame.GetPaddingX());

                // Invoke() waits for the UI thread, so the loop never grabs
                // faster than the window can paint.
                Invoke(() => ShowFrame(bitmap));
            }
        }

        private void ShowFrame(Bitmap bitmap)
        {
            Image? previous = _pictureBox.Image;
            _pictureBox.Image = bitmap;
            previous?.Dispose();
        }

        private void OnExposureScroll(object? sender, EventArgs e)
        {
            ShowExposure(_exposureBar.Value);
            Interlocked.Exchange(ref _pendingExposureUs, _exposureBar.Value);
        }

        private void ShowExposure(int exposureUs)
        {
            _exposureLabel.Text = $"Exposure time {exposureUs / 1000.0:F1} ms";
        }

        // Copies MONO8 rows, dropping the driver's line padding, into an 8-bit
        // indexed bitmap with a grayscale palette.
        private static Bitmap CreateBitmap(byte[] pixels, int width, int height, int paddingX)
        {
            var bitmap = new Bitmap(width, height, PixelFormat.Format8bppIndexed);

            ColorPalette palette = bitmap.Palette;
            for (int i = 0; i < 256; i++)
            {
                palette.Entries[i] = Color.FromArgb(i, i, i);
            }
            bitmap.Palette = palette;

            BitmapData data = bitmap.LockBits(
                new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            try
            {
                int sourceStride = width + paddingX;
                for (int y = 0; y < height; y++)
                {
                    Marshal.Copy(pixels, y * sourceStride, data.Scan0 + y * data.Stride, width);
                }
            }
            finally
            {
                bitmap.UnlockBits(data);
            }

            return bitmap;
        }
    }
}
