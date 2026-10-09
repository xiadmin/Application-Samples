' Sample name: Capture-100-images-to-file
' Category: Basic acquisition / image capture
' OS platform: Windows
' Hardware platform: Cross-platform
' API type: xiAPI.NET
' Short description: Captures 100 MONO8 images and saves them as JPEG files
' with the WPF JpegBitmapEncoder.

Imports System
Imports System.IO
Imports System.Windows.Media
Imports System.Windows.Media.Imaging
Imports xiApi.NET

Module Program

    Private Const FrameCount As Integer = 100
    Private Const ExposureUs As Integer = 2_000
    Private Const GainDb As Single = 5.0F
    Private Const GrabTimeoutMs As Integer = 5_000
    Private Const OutputDirectory As String = "images"
    Private Const JpegQualityLevel As Integer = 100
    Private Const Dpi As Double = 96.0

    ' WPF imaging objects are created on this thread, and WPF expects STA.
    <STAThread>
    Function Main() As Integer
        Dim cam As New xiCam()
        Dim isDeviceOpen As Boolean = False
        Dim isAcquiring As Boolean = False
        Dim exitCode As Integer = 0

        Try
            If cam.GetNumberDevices() = 0 Then
                Console.Error.WriteLine("Error: no XIMEA cameras detected")
                Return 1
            End If

            Directory.CreateDirectory(OutputDirectory)

            Console.WriteLine("Opening camera index 0")
            cam.OpenDevice(0)
            isDeviceOpen = True
            Console.WriteLine($"Camera {cam.GetParamString(PRM.DEVICE_NAME)}, serial number {cam.GetParamString(PRM.DEVICE_SN)}")

            cam.SetParam(PRM.EXPOSURE, ExposureUs)
            cam.SetParam(PRM.GAIN, GainDb)
            ' MONO8 works on every camera; a color sensor delivers its luminance.
            cam.SetParam(PRM.IMAGE_DATA_FORMAT, IMG_FORMAT.MONO8)
            ' With the SAFE policy xiAPI copies each frame into the array the
            ' application passes in, so one array serves every frame.
            cam.SetParam(PRM.BUFFER_POLICY, BUFF_POLICY.SAFE)

            cam.StartAcquisition()
            isAcquiring = True

            ' The payload size includes the line padding.
            Dim pixels(cam.GetParamInt(PRM.IMAGE_PAYLOAD_SIZE) - 1) As Byte
            For i As Integer = 0 To FrameCount - 1
                ' xiCam.GetImage has overloads that differ only in ByRef, which
                ' VB cannot choose between (BC31429); GetImageByteArray fills
                ' the same preallocated array.
                cam.GetImageByteArray(pixels, GrabTimeoutMs)
                Dim frame As ImgParams = cam.GetLastImageParams()

                Dim fileName As String = Path.Combine(OutputDirectory, $"image_{i:D3}.jpg")
                SaveMono8Jpeg(fileName, pixels, frame.GetWidth(), frame.GetHeight(), frame.GetPaddingX())
                Console.WriteLine($"Saved frame {i + 1}/{FrameCount} ({frame.GetWidth()}x{frame.GetHeight()}) to {fileName}")
            Next
        Catch ex As xiExc
            Console.Error.WriteLine($"Error: {ex.Message}")
            exitCode = 1
        Catch ex As IOException
            Console.Error.WriteLine($"Error: could not write JPEG file: {ex.Message}")
            exitCode = 1
        Finally
            If isAcquiring Then
                Try
                    cam.StopAcquisition()
                Catch ex As xiExc
                    Console.Error.WriteLine($"Error while stopping acquisition: {ex.Message}")
                    exitCode = 1
                End Try
            End If

            If isDeviceOpen Then
                Try
                    cam.CloseDevice()
                Catch ex As xiExc
                    Console.Error.WriteLine($"Error while closing device: {ex.Message}")
                    exitCode = 1
                End Try
            End If
        End Try

        If exitCode = 0 Then
            Console.WriteLine("Done")
        End If

        Return exitCode
    End Function

    ' Encodes an 8-bit grayscale frame as JPEG. Each line in the buffer is
    ' followed by paddingX bytes, so the stride is width + paddingX.
    Private Sub SaveMono8Jpeg(fileName As String, pixels As Byte(), width As Integer, height As Integer, paddingX As Integer)
        Dim stride As Integer = width + paddingX
        Dim image As BitmapSource = BitmapSource.Create(width, height, Dpi, Dpi, PixelFormats.Gray8, Nothing, pixels, stride)

        Dim encoder As New JpegBitmapEncoder() With {.QualityLevel = JpegQualityLevel}
        encoder.Frames.Add(BitmapFrame.Create(image))

        Using stream As New FileStream(fileName, FileMode.Create)
            encoder.Save(stream)
        End Using
    End Sub

End Module
