#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <GenTL_v1_5.h>

namespace sample
{

// Thrown for every GenTL call that does not return GC_ERR_SUCCESS.
class GenTLError : public std::runtime_error
{
public:
    GenTLError(const std::string& call, GenTL::GC_ERROR status);

    GenTL::GC_ERROR status() const { return m_status; }

private:
    GenTL::GC_ERROR m_status;
};

void checkGc(GenTL::GC_ERROR status, const char* p_call);

// Function table resolved from a GenTL producer (.cti) loaded at run time.
// Only the functions this sample calls are listed.
struct GenTLApi
{
    GenTL::PGCInitLib gcInitLib = nullptr;
    GenTL::PGCCloseLib gcCloseLib = nullptr;
    GenTL::PGCReadPort gcReadPort = nullptr;
    GenTL::PGCWritePort gcWritePort = nullptr;
    GenTL::PGCRegisterEvent gcRegisterEvent = nullptr;
    GenTL::PGCUnregisterEvent gcUnregisterEvent = nullptr;
    GenTL::PEventGetData eventGetData = nullptr;
    GenTL::PTLOpen tlOpen = nullptr;
    GenTL::PTLClose tlClose = nullptr;
    GenTL::PTLUpdateInterfaceList tlUpdateInterfaceList = nullptr;
    GenTL::PTLGetNumInterfaces tlGetNumInterfaces = nullptr;
    GenTL::PTLGetInterfaceID tlGetInterfaceID = nullptr;
    GenTL::PTLOpenInterface tlOpenInterface = nullptr;
    GenTL::PIFClose ifClose = nullptr;
    GenTL::PIFUpdateDeviceList ifUpdateDeviceList = nullptr;
    GenTL::PIFGetNumDevices ifGetNumDevices = nullptr;
    GenTL::PIFGetDeviceID ifGetDeviceID = nullptr;
    GenTL::PIFOpenDevice ifOpenDevice = nullptr;
    GenTL::PDevClose devClose = nullptr;
    GenTL::PDevGetPort devGetPort = nullptr;
    GenTL::PDevGetDataStreamID devGetDataStreamID = nullptr;
    GenTL::PDevOpenDataStream devOpenDataStream = nullptr;
    GenTL::PDSClose dsClose = nullptr;
    GenTL::PDSAnnounceBuffer dsAnnounceBuffer = nullptr;
    GenTL::PDSQueueBuffer dsQueueBuffer = nullptr;
    GenTL::PDSRevokeBuffer dsRevokeBuffer = nullptr;
    GenTL::PDSFlushQueue dsFlushQueue = nullptr;
    GenTL::PDSStartAcquisition dsStartAcquisition = nullptr;
    GenTL::PDSStopAcquisition dsStopAcquisition = nullptr;
    GenTL::PDSGetBufferInfo dsGetBufferInfo = nullptr;
};

// Loads a GenTL producer, resolves GenTLApi and initializes the library.
// The destructor closes the library and unloads the module.
class GenTLProducer
{
public:
    explicit GenTLProducer(const std::string& ctiPath);
    ~GenTLProducer();

    GenTLProducer(const GenTLProducer&) = delete;
    GenTLProducer& operator=(const GenTLProducer&) = delete;

    const GenTLApi& api() const { return m_api; }

private:
    struct ModuleCloser
    {
        void operator()(void* p_module) const;
    };

    std::unique_ptr<void, ModuleCloser> m_module;
    GenTLApi m_api;
};

// Runs a cleanup action when the scope ends. Used to release GenTL handles in
// reverse order of acquisition, also when an exception unwinds the stack.
class ScopeGuard
{
public:
    explicit ScopeGuard(std::function<void()> cleanup) : m_cleanup(std::move(cleanup)) {}
    ~ScopeGuard()
    {
        if (m_cleanup)
            m_cleanup();
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

private:
    std::function<void()> m_cleanup;
};

// Cleanup paths cannot throw, so failures there are reported on stderr.
void reportCleanupFailure(GenTL::GC_ERROR status, const char* p_call);

} // namespace sample
