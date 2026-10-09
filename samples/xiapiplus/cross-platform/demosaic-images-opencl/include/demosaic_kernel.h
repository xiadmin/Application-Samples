#pragma once

namespace sample
{

// OpenCL C source of the "demosaic" kernel, compiled at run time. It is kept
// in the executable so the sample needs no kernel file next to it.
extern const char* const demosaicKernelSource;

} // namespace sample
