#include <cstdio>
#include <cstdint>
#include <iostream>
#include <m3api/xiApi.h>
#include <cuda_runtime.h>

#ifndef _DEPACKING_HEADER_H_
    #define _DEPACKING_HEADER_H_

    int DepackBuffer(uint8_t* src_buffer, uint8_t* dst_buffer, int transport_format, int width, int height);
    
    //GPU versions

    void DepackPfncLsb10Bit(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void DepackPfncLsb12Bit(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void Depack10bitGrouping160(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void Depack12bitGrouping24(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void Depack12bitGrouping192(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    __global__ void DepackPfncLsb12BitKernel (uint8_t* src_buffer, uint8_t* dst_buffer, int index_max);

    __global__ void DepackPfncLsb10BitKernel (uint8_t* src_buffer, uint8_t* dst_buffer, int index_max);

    __global__ void Depack10bitGrouping160Kernel (uint8_t* src_buffer, uint8_t* dst_buffer, int index_max);

    __global__ void Depack12bitGrouping24Kernel (uint8_t* src_buffer, uint8_t* dst_buffer, int index_max);

    __global__ void Depack12bitGrouping192Kernel (uint8_t* src_buffer, uint8_t* dst_buffer, int index_max);

    //CPU versions

    int DepackBufferCPU(uint8_t* src_buffer, uint8_t* dst_buffer, int transport_format, int width, int height);

    void DepackPfncLsb10BitCPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void DepackPfncLsb12BitCPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void Depack10bitGrouping160CPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void Depack12bitGrouping24CPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);

    void Depack12bitGrouping192CPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height);



#endif