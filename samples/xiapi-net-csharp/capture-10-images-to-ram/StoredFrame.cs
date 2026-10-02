#nullable enable

namespace Ximea.Samples
{
    /// <summary>A frame copied out of the driver buffer, owned by the application.</summary>
    internal sealed record StoredFrame(int Width, int Height, int FrameNumber, byte[] Pixels);
}
