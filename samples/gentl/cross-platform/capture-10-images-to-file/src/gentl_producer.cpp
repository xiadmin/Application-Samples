#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "gentl_producer.h"

namespace sample
{

namespace
{

void* openModule(const std::string& path)
{
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW);
#endif
}

void closeModule(void* p_module)
{
#ifdef _WIN32
    FreeLibrary(reinterpret_cast<HMODULE>(p_module));
#else
    dlclose(p_module);
#endif
}

void* findSymbol(void* p_module, const char* p_name)
{
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(p_module), p_name));
#else
    return dlsym(p_module, p_name);
#endif
}

template <typename T>
void resolve(void* p_module, const char* p_name, T& function)
{
    function = reinterpret_cast<T>(findSymbol(p_module, p_name));
    if (function == nullptr)
        throw std::runtime_error(std::string("GenTL producer does not export ") + p_name);
}

} // namespace

GenTLError::GenTLError(const std::string& call, GenTL::GC_ERROR status)
    : std::runtime_error(call + " failed with GenTL error " + std::to_string(status)),
      m_status(status)
{
}

void checkGc(GenTL::GC_ERROR status, const char* p_call)
{
    if (status != GenTL::GC_ERR_SUCCESS)
        throw GenTLError(p_call, status);
}

void reportCleanupFailure(GenTL::GC_ERROR status, const char* p_call)
{
    if (status != GenTL::GC_ERR_SUCCESS)
        std::cerr << "Warning: " << p_call << " failed with GenTL error " << status << "\n";
}

void GenTLProducer::ModuleCloser::operator()(void* p_module) const
{
    closeModule(p_module);
}

GenTLProducer::GenTLProducer(const std::string& ctiPath) : m_module(openModule(ctiPath))
{
    if (!m_module)
        throw std::runtime_error("Could not load GenTL producer " + ctiPath);

    void* p_module = m_module.get();
    resolve(p_module, "GCInitLib", m_api.gcInitLib);
    resolve(p_module, "GCCloseLib", m_api.gcCloseLib);
    resolve(p_module, "GCReadPort", m_api.gcReadPort);
    resolve(p_module, "GCWritePort", m_api.gcWritePort);
    resolve(p_module, "GCRegisterEvent", m_api.gcRegisterEvent);
    resolve(p_module, "GCUnregisterEvent", m_api.gcUnregisterEvent);
    resolve(p_module, "EventGetData", m_api.eventGetData);
    resolve(p_module, "TLOpen", m_api.tlOpen);
    resolve(p_module, "TLClose", m_api.tlClose);
    resolve(p_module, "TLUpdateInterfaceList", m_api.tlUpdateInterfaceList);
    resolve(p_module, "TLGetNumInterfaces", m_api.tlGetNumInterfaces);
    resolve(p_module, "TLGetInterfaceID", m_api.tlGetInterfaceID);
    resolve(p_module, "TLOpenInterface", m_api.tlOpenInterface);
    resolve(p_module, "IFClose", m_api.ifClose);
    resolve(p_module, "IFUpdateDeviceList", m_api.ifUpdateDeviceList);
    resolve(p_module, "IFGetNumDevices", m_api.ifGetNumDevices);
    resolve(p_module, "IFGetDeviceID", m_api.ifGetDeviceID);
    resolve(p_module, "IFOpenDevice", m_api.ifOpenDevice);
    resolve(p_module, "DevClose", m_api.devClose);
    resolve(p_module, "DevGetPort", m_api.devGetPort);
    resolve(p_module, "DevGetDataStreamID", m_api.devGetDataStreamID);
    resolve(p_module, "DevOpenDataStream", m_api.devOpenDataStream);
    resolve(p_module, "DSClose", m_api.dsClose);
    resolve(p_module, "DSAnnounceBuffer", m_api.dsAnnounceBuffer);
    resolve(p_module, "DSQueueBuffer", m_api.dsQueueBuffer);
    resolve(p_module, "DSRevokeBuffer", m_api.dsRevokeBuffer);
    resolve(p_module, "DSFlushQueue", m_api.dsFlushQueue);
    resolve(p_module, "DSStartAcquisition", m_api.dsStartAcquisition);
    resolve(p_module, "DSStopAcquisition", m_api.dsStopAcquisition);
    resolve(p_module, "DSGetBufferInfo", m_api.dsGetBufferInfo);

    // m_module unloads the producer if this throws.
    checkGc(m_api.gcInitLib(), "GCInitLib");
}

GenTLProducer::~GenTLProducer()
{
    reportCleanupFailure(m_api.gcCloseLib(), "GCCloseLib");
}

} // namespace sample
