/*
 * Sample name: Capture-10-concatenated-images-to-file
 * Category: GenTL / concatenated images
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: GenTL
 * Short description: Captures 2 GenTL transport buffers that each hold 5
 * concatenated images, splits every buffer into its images and saves them as
 * TIFF files.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <tiffio.h>

// xiApi.h pulls in <windows.h>, so it must come before the SDK GenTL helper
// headers: they declare `using namespace std;`, which makes the Windows `byte`
// type ambiguous with std::byte in C++17.
#include <xiApi.h>
#include <xi_GenTL.h>

namespace fs = std::filesystem;

static constexpr int imagesPerBuffer = 5;
static constexpr int bufferCount = 2;
static constexpr int exposureUs = 10000;
static constexpr int gainDb = 0;
static constexpr std::uint64_t bufferTimeoutMs = 10000;

struct Options
{
    fs::path outputDir = ".";
};

static void checkXi(XI_RETURN status, const char* p_call)
{
    if (status != XI_OK)
        throw std::runtime_error(std::string(p_call) + " returned xiAPI error " + std::to_string(status));
}

// Opens the first camera and closes it when the object goes out of scope.
class XiCamera
{
public:
    XiCamera()
    {
        DWORD cameraCount = 0;
        checkXi(xiGetNumberDevices(&cameraCount), "xiGetNumberDevices");
        if (cameraCount == 0)
            throw std::runtime_error("no XIMEA cameras detected");
        checkXi(xiOpenDevice(0, &m_handle), "xiOpenDevice");
    }

    ~XiCamera()
    {
        const XI_RETURN status = xiCloseDevice(m_handle);
        if (status != XI_OK)
            std::cerr << "Warning: xiCloseDevice returned xiAPI error " << status << "\n";
    }

    XiCamera(const XiCamera&) = delete;
    XiCamera& operator=(const XiCamera&) = delete;

    HANDLE handle() const { return m_handle; }

private:
    HANDLE m_handle = nullptr;
};

// Runs a cleanup action when the scope ends, also during stack unwinding.
// The GenTL helpers throw on failure, and a destructor must not, so failures
// are reported on stderr instead.
class ScopeGuard
{
public:
    explicit ScopeGuard(std::function<void()> cleanup) : m_cleanup(std::move(cleanup)) {}
    ~ScopeGuard()
    {
        try
        {
            m_cleanup();
        }
        catch (const std::exception& error)
        {
            std::cerr << "Warning: cleanup failed: " << error.what() << "\n";
        }
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

private:
    std::function<void()> m_cleanup;
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--output-dir <dir>]\n"
              << "  --output-dir  Directory for the TIFF files. Default: current directory.\n";
}

static std::optional<Options> parseArguments(int argc, char* argv[], bool& isHelpRequested)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h")
        {
            isHelpRequested = true;
            return std::nullopt;
        }
        if (argument == "--output-dir" && i + 1 < argc)
            options.outputDir = argv[++i];
        else
        {
            std::cerr << "Error: unknown or incomplete argument '" << argument << "'\n";
            return std::nullopt;
        }
    }
    return options;
}

// Writes one 8-bit single-channel image as an uncompressed TIFF file.
static void writeMono8Tiff(const fs::path& path, const std::uint8_t* p_pixels, std::uint32_t width,
                           std::uint32_t height)
{
    const std::string pathText = path.string();
    std::unique_ptr<TIFF, decltype(&TIFFClose)> p_tiff(TIFFOpen(pathText.c_str(), "w"), &TIFFClose);
    if (!p_tiff)
        throw std::runtime_error("TIFFOpen failed for " + pathText);

    const bool isHeaderWritten =
        TIFFSetField(p_tiff.get(), TIFFTAG_IMAGEWIDTH, width) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_IMAGELENGTH, height) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_SAMPLESPERPIXEL, static_cast<std::uint16_t>(1)) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_BITSPERSAMPLE, static_cast<std::uint16_t>(8)) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_COMPRESSION, COMPRESSION_NONE) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_ROWSPERSTRIP, height) == 1;
    if (!isHeaderWritten)
        throw std::runtime_error("TIFFSetField failed for " + pathText);

    for (std::uint32_t row = 0; row < height; ++row)
    {
        // TIFFWriteScanline() takes a non-const pointer but does not modify the row.
        auto* p_row = const_cast<std::uint8_t*>(p_pixels + static_cast<std::size_t>(row) * width);
        if (TIFFWriteScanline(p_tiff.get(), p_row, row, 0) < 0)
            throw std::runtime_error("TIFFWriteScanline failed for " + pathText);
    }
}

// Splits one transport buffer into its concatenated images and saves each one.
static void saveConcatenatedImages(ImgBuffer& buffer, int bufferIndex, std::size_t payloadSize,
                                   std::size_t imageOffset, const fs::path& outputDir)
{
    if (!buffer.HasNewData())
        throw std::runtime_error("frame transfer was interrupted, the buffer holds no valid images");

    void* p_base = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;
    buffer.GetInfoPtr(GenTL::BUFFER_INFO_BASE, &p_base);
    buffer.GetInfoSizet(GenTL::BUFFER_INFO_WIDTH, &width);
    buffer.GetInfoSizet(GenTL::BUFFER_INFO_HEIGHT, &height);

    // RAW8 transport data carries one byte per pixel. Every image starts
    // imageOffset bytes after the previous one, and the last one must still
    // end inside the buffer.
    const std::size_t imageBytes = width * height;
    const std::size_t requiredBytes = imageOffset * (imagesPerBuffer - 1) + imageBytes;
    if (p_base == nullptr || imageBytes == 0 || imageOffset < imageBytes || requiredBytes > payloadSize)
        throw std::runtime_error("the camera device reported a buffer layout that does not hold " +
                                 std::to_string(imagesPerBuffer) + " images");

    const auto* p_bytes = static_cast<const std::uint8_t*>(p_base);
    for (int i = 0; i < imagesPerBuffer; ++i)
    {
        char fileName[64];
        std::snprintf(fileName, sizeof(fileName), "buffer%d_image%d.tiff", bufferIndex + 1, i + 1);
        const fs::path filePath = outputDir / fileName;
        writeMono8Tiff(filePath, p_bytes + i * imageOffset, static_cast<std::uint32_t>(width),
                       static_cast<std::uint32_t>(height));
        std::cout << "Saved " << filePath.string() << "\n";
    }
}

static int run(const Options& options)
{
    std::error_code directoryError;
    fs::create_directories(options.outputDir, directoryError);
    if (directoryError)
    {
        std::cerr << "Error: cannot create " << options.outputDir << ": " << directoryError.message() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Opening first camera\n";
    XiCamera camera;
    const HANDLE cameraHandle = camera.handle();

    checkXi(xiSetParamInt(cameraHandle, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8), "xiSetParamInt(IMAGE_DATA_FORMAT)");
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA),
            "xiSetParamInt(IMAGE_DATA_FORMAT)");

    // With concatenation the camera packs several frames into one transport
    // buffer, so the payload size and the image offset are read afterwards.
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_CONCAT_IMG_COUNT, imagesPerBuffer), "xiSetParamInt(CONCAT_IMG_COUNT)");
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_CONCAT_IMG_MODE, XI_ON), "xiSetParamInt(CONCAT_IMG_MODE)");
    int payloadSize = 0;
    checkXi(xiGetParamInt(cameraHandle, XI_PRM_IMAGE_PAYLOAD_SIZE, &payloadSize), "xiGetParamInt(IMAGE_PAYLOAD_SIZE)");
    int imageOffset = 0;
    checkXi(xiGetParamInt(cameraHandle, XI_PRM_CONCAT_IMG_TRANSPORT_IMG_OFFSET, &imageOffset),
            "xiGetParamInt(CONCAT_IMG_TRANSPORT_IMG_OFFSET)");
    std::cout << imagesPerBuffer << " images per buffer, payload " << payloadSize << " bytes, image offset "
              << imageOffset << " bytes\n";

    checkXi(xiSetParamInt(cameraHandle, XI_PRM_GENTL_DATASTREAM_ENABLED, XI_ON),
            "xiSetParamInt(GENTL_DATASTREAM_ENABLED)");
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_EXPOSURE, exposureUs), "xiSetParamInt(EXPOSURE)");
    checkXi(xiSetParamFloat(cameraHandle, XI_PRM_GAIN, static_cast<float>(gainDb)), "xiSetParamFloat(GAIN)");

    GenTL::DS_HANDLE streamHandle = nullptr;
    checkXi(xiGetParamString(cameraHandle, XI_PRM_GENTL_DATASTREAM_CONTEXT, &streamHandle, sizeof(streamHandle)),
            "xiGetParamString(GENTL_DATASTREAM_CONTEXT)");

    // xiGenTL is the higher-level SDK helper: it owns the buffer list that the
    // producer allocates and matches each new-buffer event to an ImgBuffer.
    xiGenTL genTl;
    genTl.Initialize(streamHandle);
    ScopeGuard uninitialize([&] { genTl.Uninitialize(); });

    std::cout << "Preparing " << bufferCount << " acquisition buffers\n";
    ScopeGuard revokeBuffers([&] {
        genTl.FlushQueue(GenTL::ACQ_QUEUE_ALL_DISCARD);
        genTl.RevokeAllBuffers();
    });
    genTl.AllocAndAnnounceAllBuffers(static_cast<std::size_t>(payloadSize), bufferCount);
    genTl.QueueAllBuffers();

    std::cout << "Starting acquisition\n";
    genTl.StartAcquisition();
    ScopeGuard abortAcquisition([&] { genTl.AbortAcquisition(); });

    for (int i = 0; i < bufferCount; ++i)
    {
        ImgBuffer buffer;
        genTl.GetNextBuffer(buffer, bufferTimeoutMs);
        saveConcatenatedImages(buffer, i, static_cast<std::size_t>(payloadSize), static_cast<std::size_t>(imageOffset),
                               options.outputDir);
        genTl.QueueBuffer(buffer);
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    bool isHelpRequested = false;
    const std::optional<Options> options = parseArguments(argc, argv, isHelpRequested);
    if (!options)
    {
        printUsage(argv[0]);
        return isHelpRequested ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    try
    {
        return run(*options);
    }
    catch (const std::exception& error)
    {
        // GenTL_Exception from the SDK helpers also derives from std::exception.
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
