/*
 * Sample name: Capture-10-images-to-file
 * Category: GenTL / low-level acquisition
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: GenTL
 * Short description: Loads the XIMEA GenTL producer (.cti) at run time, opens
 * the first camera through the GenTL module hierarchy, captures 10 images
 * into application buffers and saves them as TIFF files.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <GenTL_v1_5.h>

#include "gentl_producer.h"
#include "tiff_writer.h"

namespace fs = std::filesystem;

static constexpr int frameCount = 10;
static constexpr int bufferCount = 10;
static constexpr std::size_t bufferAlignment = 4096;
static constexpr std::uint64_t frameTimeoutMs = 2000;
static constexpr std::uint64_t discoveryTimeoutMs = 5000;
static constexpr float exposureUs = 1000.0f;

// XIMEA producers expose camera features at fixed port addresses. The legacy
// producer (ximea.gentl.cti) and the current one (ximea.gentl2.cti) use
// different register maps; a generic consumer would read the addresses from
// the device's GenICam XML instead.
struct RegisterMap
{
    std::uint64_t width;
    std::uint64_t height;
    std::uint64_t payloadSize;
    std::uint64_t exposureTime;
    std::uint64_t acquisitionStart; // 0 when the producer starts the camera itself
    std::uint64_t acquisitionStop;
};

static constexpr RegisterMap legacyRegisterMap{0x102100, 0x102200, 0x106600, 0x103400, 0, 0};
static constexpr RegisterMap currentRegisterMap{0x200218, 0x200220, 0x200660, 0x200340, 0x200310, 0x200320};

struct Options
{
    std::string ctiPath;
    fs::path outputDir = ".";
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--cti <producer.cti>] [--output-dir <dir>]\n"
              << "  --cti         GenTL producer to load. Default: ximea.gentl2.cti or\n"
              << "                ximea.gentl.cti from GENICAM_GENTL64_PATH (GENICAM_GENTL32_PATH\n"
              << "                in 32-bit builds).\n"
              << "  --output-dir  Directory for the TIFF files. Default: current directory.\n";
}

static std::optional<Options> parseArguments(int argc, char* argv[], bool& isHelpRequested)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        const bool hasValue = i + 1 < argc;
        if (argument == "--help" || argument == "-h")
        {
            isHelpRequested = true;
            return std::nullopt;
        }
        if (argument == "--cti" && hasValue)
            options.ctiPath = argv[++i];
        else if (argument == "--output-dir" && hasValue)
            options.outputDir = argv[++i];
        else
        {
            std::cerr << "Error: unknown or incomplete argument '" << argument << "'\n";
            return std::nullopt;
        }
    }
    return options;
}

// Looks for a XIMEA producer in the directories listed by the standard GenTL
// environment variable. The current producer is preferred over the legacy one.
static std::string findXimeaProducer()
{
    const char* p_variable = sizeof(void*) == 8 ? "GENICAM_GENTL64_PATH" : "GENICAM_GENTL32_PATH";
    const char* p_searchPath = std::getenv(p_variable);
    if (p_searchPath == nullptr)
        return {};

#ifdef _WIN32
    const char separator = ';';
#else
    const char separator = ':';
#endif

    for (const char* p_fileName : {"ximea.gentl2.cti", "ximea.gentl.cti"})
    {
        std::stringstream directories(p_searchPath);
        std::string directory;
        while (std::getline(directories, directory, separator))
        {
            if (directory.empty())
                continue;
            const fs::path candidate = fs::path(directory) / p_fileName;
            std::error_code error;
            if (fs::is_regular_file(candidate, error))
                return candidate.string();
        }
    }
    return {};
}

static const RegisterMap& registerMapFor(const std::string& ctiPath)
{
    const std::string fileName = fs::path(ctiPath).filename().string();
    return fileName.find("gentl2") != std::string::npos ? currentRegisterMap : legacyRegisterMap;
}

static std::uint32_t readRegister(const sample::GenTLApi& api, GenTL::PORT_HANDLE port, std::uint64_t address)
{
    std::uint32_t value = 0;
    std::size_t size = sizeof(value);
    sample::checkGc(api.gcReadPort(port, address, &value, &size), "GCReadPort");
    return value;
}

// Writes a XIMEA command register. The producer expects a size_t-sized write.
static GenTL::GC_ERROR writeCommand(const sample::GenTLApi& api, GenTL::PORT_HANDLE port, std::uint64_t address,
                                    std::size_t value)
{
    std::size_t size = sizeof(value);
    return api.gcWritePort(port, address, &value, &size);
}

static int run(const Options& options)
{
    const std::string ctiPath = options.ctiPath.empty() ? findXimeaProducer() : options.ctiPath;
    if (ctiPath.empty())
    {
        std::cerr << "Error: no XIMEA GenTL producer found, so no camera device can be opened. "
                     "Set GENICAM_GENTL64_PATH or pass --cti <producer.cti>.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Loading GenTL producer " << ctiPath << "\n";
    sample::GenTLProducer producer(ctiPath);
    const sample::GenTLApi& api = producer.api();
    const RegisterMap& registers = registerMapFor(ctiPath);

    GenTL::TL_HANDLE system = nullptr;
    sample::checkGc(api.tlOpen(&system), "TLOpen");
    sample::ScopeGuard closeSystem([&] { sample::reportCleanupFailure(api.tlClose(system), "TLClose"); });
    sample::checkGc(api.tlUpdateInterfaceList(system, nullptr, discoveryTimeoutMs), "TLUpdateInterfaceList");

    // Open the first interface that reports at least one device.
    std::uint32_t interfaceCount = 0;
    sample::checkGc(api.tlGetNumInterfaces(system, &interfaceCount), "TLGetNumInterfaces");
    GenTL::IF_HANDLE interfaceHandle = nullptr;
    for (std::uint32_t i = 0; i < interfaceCount && interfaceHandle == nullptr; ++i)
    {
        char interfaceId[256] = "";
        std::size_t size = sizeof(interfaceId);
        sample::checkGc(api.tlGetInterfaceID(system, i, interfaceId, &size), "TLGetInterfaceID");

        GenTL::IF_HANDLE candidate = nullptr;
        sample::checkGc(api.tlOpenInterface(system, interfaceId, &candidate), "TLOpenInterface");
        std::uint32_t deviceCount = 0;
        const GenTL::GC_ERROR updateStatus = api.ifUpdateDeviceList(candidate, nullptr, discoveryTimeoutMs);
        const GenTL::GC_ERROR countStatus = api.ifGetNumDevices(candidate, &deviceCount);
        if (updateStatus == GenTL::GC_ERR_SUCCESS && countStatus == GenTL::GC_ERR_SUCCESS && deviceCount > 0)
            interfaceHandle = candidate;
        else
            sample::reportCleanupFailure(api.ifClose(candidate), "IFClose");
    }
    if (interfaceHandle == nullptr)
    {
        std::cerr << "Error: the GenTL producer found no camera device\n";
        return EXIT_FAILURE;
    }
    sample::ScopeGuard closeInterface([&] { sample::reportCleanupFailure(api.ifClose(interfaceHandle), "IFClose"); });

    char deviceId[256] = "";
    std::size_t deviceIdSize = sizeof(deviceId);
    sample::checkGc(api.ifGetDeviceID(interfaceHandle, 0, deviceId, &deviceIdSize), "IFGetDeviceID");
    std::cout << "Opening device " << deviceId << "\n";

    GenTL::DEV_HANDLE device = nullptr;
    sample::checkGc(api.ifOpenDevice(interfaceHandle, deviceId, GenTL::DEVICE_ACCESS_EXCLUSIVE, &device),
                    "IFOpenDevice");
    sample::ScopeGuard closeDevice([&] { sample::reportCleanupFailure(api.devClose(device), "DevClose"); });

    GenTL::PORT_HANDLE port = nullptr;
    sample::checkGc(api.devGetPort(device, &port), "DevGetPort");

    const std::uint32_t width = readRegister(api, port, registers.width);
    const std::uint32_t height = readRegister(api, port, registers.height);
    const std::uint32_t payloadSize = readRegister(api, port, registers.payloadSize);
    std::cout << "Resolution " << width << "x" << height << ", payload " << payloadSize << " bytes\n";

    // The sample saves 8-bit monochrome TIFFs, so the payload must hold at
    // least one byte per pixel.
    const std::size_t imageBytes = static_cast<std::size_t>(width) * height;
    if (width == 0 || height == 0 || payloadSize < imageBytes)
    {
        std::cerr << "Error: the camera device reports an unsupported image size or pixel format\n";
        return EXIT_FAILURE;
    }

    float exposure = exposureUs;
    std::size_t exposureSize = sizeof(exposure);
    sample::checkGc(api.gcWritePort(port, registers.exposureTime, &exposure, &exposureSize), "GCWritePort(exposure)");

    char streamId[256] = "";
    std::size_t streamIdSize = sizeof(streamId);
    sample::checkGc(api.devGetDataStreamID(device, 0, streamId, &streamIdSize), "DevGetDataStreamID");
    GenTL::DS_HANDLE stream = nullptr;
    sample::checkGc(api.devOpenDataStream(device, streamId, &stream), "DevOpenDataStream");
    sample::ScopeGuard closeStream([&] { sample::reportCleanupFailure(api.dsClose(stream), "DSClose"); });

    // Each buffer is over-allocated so std::align can place a 4 KiB-aligned
    // payload inside it; the transport layer requires aligned memory.
    std::vector<std::vector<std::uint8_t>> storage(bufferCount);
    std::vector<GenTL::BUFFER_HANDLE> buffers;
    sample::ScopeGuard revokeBuffers([&] {
        sample::reportCleanupFailure(api.dsFlushQueue(stream, GenTL::ACQ_QUEUE_ALL_DISCARD), "DSFlushQueue");
        for (GenTL::BUFFER_HANDLE buffer : buffers)
            sample::reportCleanupFailure(api.dsRevokeBuffer(stream, buffer, nullptr, nullptr), "DSRevokeBuffer");
    });
    for (auto& bytes : storage)
    {
        bytes.resize(payloadSize + bufferAlignment);
        void* p_aligned = bytes.data();
        std::size_t space = bytes.size();
        std::align(bufferAlignment, payloadSize, p_aligned, space);

        GenTL::BUFFER_HANDLE buffer = nullptr;
        sample::checkGc(api.dsAnnounceBuffer(stream, p_aligned, payloadSize, nullptr, &buffer), "DSAnnounceBuffer");
        buffers.push_back(buffer);
        sample::checkGc(api.dsQueueBuffer(stream, buffer), "DSQueueBuffer");
    }

    GenTL::EVENT_HANDLE newBufferEvent = nullptr;
    sample::checkGc(api.gcRegisterEvent(stream, GenTL::EVENT_NEW_BUFFER, &newBufferEvent), "GCRegisterEvent");
    sample::ScopeGuard unregisterEvent([&] {
        sample::reportCleanupFailure(api.gcUnregisterEvent(stream, GenTL::EVENT_NEW_BUFFER), "GCUnregisterEvent");
    });

    sample::checkGc(api.dsStartAcquisition(stream, GenTL::ACQ_START_FLAGS_DEFAULT, GENTL_INFINITE),
                    "DSStartAcquisition");
    sample::ScopeGuard stopAcquisition([&] {
        if (registers.acquisitionStop != 0)
            sample::reportCleanupFailure(writeCommand(api, port, registers.acquisitionStop, 0), "AcquisitionStop");
        sample::reportCleanupFailure(api.dsStopAcquisition(stream, GenTL::ACQ_STOP_FLAGS_DEFAULT), "DSStopAcquisition");
    });
    if (registers.acquisitionStart != 0)
        sample::checkGc(writeCommand(api, port, registers.acquisitionStart, 1), "AcquisitionStart");

    std::error_code directoryError;
    fs::create_directories(options.outputDir, directoryError);
    if (directoryError)
    {
        std::cerr << "Error: cannot create " << options.outputDir << ": " << directoryError.message() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Acquiring " << frameCount << " images\n";
    int savedCount = 0;
    for (int i = 0; i < frameCount; ++i)
    {
        GenTL::EVENT_NEW_BUFFER_DATA eventData{};
        std::size_t eventSize = sizeof(eventData);
        const GenTL::GC_ERROR eventStatus = api.eventGetData(newBufferEvent, &eventData, &eventSize, frameTimeoutMs);
        if (eventStatus != GenTL::GC_ERR_SUCCESS)
        {
            std::cerr << "Error: no image " << i + 1 << " from the camera (GenTL error " << eventStatus << ")\n";
            continue;
        }

        void* p_data = nullptr;
        GenTL::INFO_DATATYPE type = GenTL::INFO_DATATYPE_PTR;
        std::size_t infoSize = sizeof(p_data);
        sample::checkGc(api.dsGetBufferInfo(stream, eventData.BufferHandle, GenTL::BUFFER_INFO_BASE, &type, &p_data,
                                            &infoSize),
                        "DSGetBufferInfo");

        char fileName[32];
        std::snprintf(fileName, sizeof(fileName), "image_%02d.tiff", i);
        const fs::path filePath = options.outputDir / fileName;
        sample::writeMono8Tiff(filePath.string(), static_cast<const std::uint8_t*>(p_data), width, height);
        std::cout << "Saved " << filePath.string() << "\n";
        ++savedCount;

        sample::checkGc(api.dsQueueBuffer(stream, eventData.BufferHandle), "DSQueueBuffer");
    }

    if (savedCount != frameCount)
    {
        std::cerr << "Error: saved " << savedCount << " of " << frameCount << " images\n";
        return EXIT_FAILURE;
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
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
