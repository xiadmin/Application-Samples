#include "depacking.h"

/**
 * @brief This function parses @param transport_format variable and calls the appropriate CPU depacking function.
 * @param src_buffer pointer to the source buffer, containg raw 10bpp or 12bpp packed picture
 * @param dst_buffer pointer to the destination buffer, set up for up to 16bpp single channel picture 
 * @param transport_format XI_GenTL_Image_Format extracted from the camera
 * @param width width of the raw, unpacked picture
 * @param height height of the rawm unpacked picture
 */
int DepackBufferCPU(uint8_t* src_buffer, uint8_t* dst_buffer, int transport_format, int width, int height)
{

    if (transport_format == XI_GenTL_Image_Format_BayerBG12p || transport_format == XI_GenTL_Image_Format_BayerGB12p ||
        transport_format == XI_GenTL_Image_Format_BayerGR12p || transport_format == XI_GenTL_Image_Format_BayerRG12p)
    {
        std::cout<<"CPU PFNC LSB 12bit depacking!"<<std::endl;
        DepackPfncLsb12BitCPU(src_buffer,dst_buffer,width,height);
    }
    else if (transport_format == XI_GenTL_Image_Format_BayerBG10p || transport_format == XI_GenTL_Image_Format_BayerGB10p ||
             transport_format == XI_GenTL_Image_Format_BayerGR10p || transport_format == XI_GenTL_Image_Format_BayerRG10p)
    {
        std::cout<<"CPU PFNC LSB 10bit depacking!"<<std::endl;
        DepackPfncLsb10BitCPU(src_buffer,dst_buffer,width,height);
    }
    else if (transport_format == XI_GenTL_Image_Format_xiBayerBG10g160 || transport_format == XI_GenTL_Image_Format_xiBayerGB10g160 ||
             transport_format == XI_GenTL_Image_Format_xiBayerGR10g160 || transport_format == XI_GenTL_Image_Format_xiBayerRG10g160)
    {
        std::cout<<"CPU 10g160 depacking!"<<std::endl;
        Depack10bitGrouping160CPU(src_buffer,dst_buffer,width,height);
    }
    else if (transport_format == XI_GenTL_Image_Format_xiBayerRG12g24 || transport_format == XI_GenTL_Image_Format_xiBayerGR12g24 ||

             transport_format == XI_GenTL_Image_Format_xiBayerGB12g24 || transport_format == XI_GenTL_Image_Format_xiBayerBG12g24)
    {
        std::cout<<"CPU 12g24 depacking!"<<std::endl;
        Depack12bitGrouping24CPU(src_buffer,dst_buffer,width,height);
    }
    else if (transport_format == XI_GenTL_Image_Format_xiBayerBG12g192 || transport_format == XI_GenTL_Image_Format_xiBayerGB12g192  ||
             transport_format == XI_GenTL_Image_Format_xiBayerGR12g192 || transport_format == XI_GenTL_Image_Format_xiBayerRG12g192)
    {
        std::cout<<"CPU 12g192 depacking!"<<std::endl;
        Depack12bitGrouping192CPU(src_buffer,dst_buffer,width,height);

    }
    else
    {
        std::cerr << "Error: Unsupported packing type!" << std::endl;
        return -1;
    }

    return 0;
}

/**
 * @brief function for depacking 10p picture on the CPU
 * @param src_buffer pointer to the source buffer, containg raw 10bpp or 12bpp packed picture
 * @param dst_buffer pointer to the destination buffer, set up for up to 16bpp single channel picture 
 * @param transport_format XI_GenTL_Image_Format extracted from the camera
 * @param width width of the raw, unpacked picture
 * @param height height of the raw unpacked picture
 */
void DepackPfncLsb10BitCPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height)
{
    for (int i = 0; i < width*height/4; i++)
    {
        int src_i = i*5;
        int dst_i = i*8;


        //four pixels of two bytes each are in five bytes
        dst_buffer[dst_i] = src_buffer[src_i];
        dst_buffer[dst_i+1] = src_buffer[src_i+1] &  0b00000011;

        dst_buffer[dst_i+2] = (src_buffer[src_i+1] & 0b11111100) >> 2 | (src_buffer[src_i+2] & 0b00000011) << 6;
        dst_buffer[dst_i+3] = (src_buffer[src_i+2] & 0b00001100) >> 2; 

        dst_buffer[dst_i+4] = (src_buffer[src_i+2] & 0b11110000) >> 4 | (src_buffer[src_i+3] & 0b00001111) << 4;
        dst_buffer[dst_i+5] = (src_buffer[src_i+3] & 0b00110000) >> 4; 
            
        dst_buffer[dst_i+6] = (src_buffer[src_i+3] & 0b11000000) >> 6 | (src_buffer[src_i+4] & 0b00111111) << 6;
        dst_buffer[dst_i+7] = (src_buffer[src_i+4] & 0b11000000) >> 6; 
    }
}

/**
 * @brief function for depacking 12p picture on the CPU
 * @param src_buffer pointer to the source buffer, containg raw 10bpp or 12bpp packed picture
 * @param dst_buffer pointer to the destination buffer, set up for up to 16bpp single channel picture 
 * @param transport_format XI_GenTL_Image_Format extracted from the camera
 * @param width width of the raw, unpacked picture
 * @param height height of the raw unpacked picture
 */
void DepackPfncLsb12BitCPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height)
{
    for (int i = 0; i < width*height/2; i++)
    {
        int src_i = i*3;
        int dst_i = i*4;
        
        //two pixels of two bytes each are in three bytes
        dst_buffer[dst_i] = src_buffer[src_i];
        dst_buffer[dst_i+1] = src_buffer[src_i+1] &  0b00001111;

        dst_buffer[dst_i+2] = (src_buffer[src_i+1] & 0b11110000) >> 4 | (src_buffer[src_i+2] & 0b00001111) << 4;
        dst_buffer[dst_i+3] = (src_buffer[src_i+2] & 0b11110000) >> 4;        
    }
    
}

/**
 * @brief function for depacking 10g160 picture on the CPU
 * @param src_buffer pointer to the source buffer, containg raw 10bpp or 12bpp packed picture
 * @param dst_buffer pointer to the destination buffer, set up for up to 16bpp single channel picture 
 * @param transport_format XI_GenTL_Image_Format extracted from the camera
 * @param width width of the raw, unpacked picture
 * @param height height of the raw unpacked picture
 */
void Depack10bitGrouping160CPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height)
{
  
    for (int i = 0; i < width*height/16; i++)
    {

        int src_i = i*20;
        int dst_i = i*16*2;
        
        //sixteen pixels of two bytes each are in twenty bytes
        dst_buffer[dst_i] = (src_buffer[src_i] & 0b00111111) << 2 | src_buffer[src_i+16] & 0b00000011 ;
        dst_buffer[dst_i+1] = (src_buffer[src_i] &  0b11000000) >> 6;

        dst_buffer[dst_i+2] = (src_buffer[src_i+1] & 0b00111111) << 2 | (src_buffer[src_i+16] & 0b00001100) >> 2;
        dst_buffer[dst_i+3] = (src_buffer[src_i+1] &  0b11000000) >> 6;

        dst_buffer[dst_i+4] = (src_buffer[src_i+2] & 0b00111111) << 2 | (src_buffer[src_i+16] & 0b00110000) >> 4;
        dst_buffer[dst_i+5] = (src_buffer[src_i+2] &  0b11000000) >> 6;

        dst_buffer[dst_i+6] = (src_buffer[src_i+3] & 0b00111111) << 2 | (src_buffer[src_i+16] & 0b11000000) >> 6;
        dst_buffer[dst_i+7] = (src_buffer[src_i+3] &  0b11000000) >> 6;

        dst_buffer[dst_i+8] = (src_buffer[src_i+4] & 0b00111111) << 2 | src_buffer[src_i+17] & 0b00000011;
        dst_buffer[dst_i+9] = (src_buffer[src_i+4] &  0b11000000) >> 6;

        dst_buffer[dst_i+10] = (src_buffer[src_i+5] & 0b00111111) << 2 | (src_buffer[src_i+17] & 0b00001100) >> 2;
        dst_buffer[dst_i+11] = (src_buffer[src_i+5] &  0b11000000) >> 6;

        dst_buffer[dst_i+12] = (src_buffer[src_i+6] & 0b00111111) << 2 | (src_buffer[src_i+17] & 0b00110000) >> 4;
        dst_buffer[dst_i+13] = (src_buffer[src_i+6] &  0b11000000) >> 6;

        dst_buffer[dst_i+14] = (src_buffer[src_i+7] & 0b00111111) << 2 | (src_buffer[src_i+17] & 0b11000000) >> 6;
        dst_buffer[dst_i+15] = (src_buffer[src_i+7] &  0b11000000) >> 6;

        dst_buffer[dst_i+16] = (src_buffer[src_i+8] & 0b00111111) << 2 | src_buffer[src_i+18] & 0b00000011;
        dst_buffer[dst_i+17] = (src_buffer[src_i+8] &  0b11000000) >> 6;

        dst_buffer[dst_i+18] = (src_buffer[src_i+9] & 0b00111111) << 2 | (src_buffer[src_i+18] & 0b00001100) >> 2;
        dst_buffer[dst_i+19] = (src_buffer[src_i+9] &  0b11000000) >> 6;

        dst_buffer[dst_i+20] = (src_buffer[src_i+10] & 0b00111111) << 2 | (src_buffer[src_i+18] & 0b00110000) << 4;
        dst_buffer[dst_i+21] = (src_buffer[src_i+10] &  0b11000000) >> 6;

        dst_buffer[dst_i+22] = (src_buffer[src_i+11] & 0b00111111) << 2 | (src_buffer[src_i+18] & 0b11000000) >> 6;
        dst_buffer[dst_i+23] = (src_buffer[src_i+11] &  0b11000000) >> 6;

        dst_buffer[dst_i+24] = (src_buffer[src_i+12] & 0b00111111) << 2 | src_buffer[src_i+19] & 0b00000011;
        dst_buffer[dst_i+25] = (src_buffer[src_i+12] &  0b11000000) >> 6;

        dst_buffer[dst_i+26] = (src_buffer[src_i+13] & 0b00111111) << 2 | (src_buffer[src_i+19] & 0b00001100) >> 2;
        dst_buffer[dst_i+27] = (src_buffer[src_i+13] &  0b11000000) >> 6;

        dst_buffer[dst_i+28] = (src_buffer[src_i+14] & 0b00111111) << 2 | (src_buffer[src_i+19] & 0b00110000) >> 4;
        dst_buffer[dst_i+29] = (src_buffer[src_i+14] &  0b11000000) >> 6;

        dst_buffer[dst_i+30] = (src_buffer[src_i+15] & 0b00111111) << 2 | (src_buffer[src_i+19] & 0b11000000) >> 6;
        dst_buffer[dst_i+31] = (src_buffer[src_i+15] &  0b11000000) >> 6;

    }

}
       
/**
 * @brief function for depacking 12g24 picture on the CPU
 * @param src_buffer pointer to the source buffer, containg raw 10bpp or 12bpp packed picture
 * @param dst_buffer pointer to the destination buffer, set up for up to 16bpp single channel picture 
 * @param transport_format XI_GenTL_Image_Format extracted from the camera
 * @param width width of the raw, unpacked picture
 * @param height height of the raw unpacked picture
 */
void Depack12bitGrouping24CPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height)
{
    for (int i = 0; i < width*height/2; i++)
    {
        int src_i = i*3;
        int dst_i = i*2*2;

    //two pixels of two bytes each are in three bytes
    dst_buffer[dst_i] = (src_buffer[src_i] & 0b00001111) << 4 | (src_buffer[src_i+2] & 0b00001111);
    dst_buffer[dst_i+1] = (src_buffer[src_i] & 0b11110000) >> 4;

    dst_buffer[dst_i+2] = (src_buffer[src_i+1] & 0b00001111) << 4 | (src_buffer[src_i+2] & 0b11110000) >> 4;
    dst_buffer[dst_i+3] = (src_buffer[src_i+1] & 0b11110000) >> 4;

    }
}

/**
 * @brief function for depacking 12g192 picture on the CPU
 * @param src_buffer pointer to the source buffer, containg raw 10bpp or 12bpp packed picture
 * @param dst_buffer pointer to the destination buffer, set up for up to 16bpp single channel picture 
 * @param transport_format XI_GenTL_Image_Format extracted from the camera
 * @param width width of the raw, unpacked picture
 * @param height height of the raw unpacked picture
 */
void Depack12bitGrouping192CPU(uint8_t* src_buffer, uint8_t* dst_buffer, int width, int height)
{
    for (int i = 0; i < width*height/16; i++)
    {
        int src_i = i*24;
        int dst_i = i*2*16;

        //two pixels of two bytes each are in three bytes
        dst_buffer[dst_i] = (src_buffer[src_i] & 0b00001111) << 4 | (src_buffer[src_i+16] & 0b00001111);
        dst_buffer[dst_i+1] = (src_buffer[src_i] & 0b11110000) >> 4;

        dst_buffer[dst_i+2] = (src_buffer[src_i+1] & 0b00001111) << 4 | (src_buffer[src_i+16] & 0b11110000)>> 4;
        dst_buffer[dst_i+3] = (src_buffer[src_i+1] & 0b11110000) >> 4;

        dst_buffer[dst_i+4] = (src_buffer[src_i+2] & 0b00001111) << 4 | (src_buffer[src_i+17] & 0b00001111);
        dst_buffer[dst_i+5] = (src_buffer[src_i+2] & 0b11110000) >> 4;

        dst_buffer[dst_i+6] = (src_buffer[src_i+3] & 0b00001111) << 4 | (src_buffer[src_i+17] & 0b11110000)>> 4;
        dst_buffer[dst_i+7] = (src_buffer[src_i+3] & 0b11110000) >> 4;

        dst_buffer[dst_i+8] = (src_buffer[src_i+4] & 0b00001111) << 4 | (src_buffer[src_i+18] & 0b00001111);
        dst_buffer[dst_i+9] = (src_buffer[src_i+4] & 0b11110000) >> 4;

        dst_buffer[dst_i+10] = (src_buffer[src_i+5] & 0b00001111) << 4 | (src_buffer[src_i+18] & 0b11110000)>> 4;
        dst_buffer[dst_i+11] = (src_buffer[src_i+5] & 0b11110000) >> 4;

        dst_buffer[dst_i+12] = (src_buffer[src_i+6] & 0b00001111) << 4 | (src_buffer[src_i+19] & 0b00001111);
        dst_buffer[dst_i+13] = (src_buffer[src_i+6] & 0b11110000) >> 4;

        dst_buffer[dst_i+14] = (src_buffer[src_i+7] & 0b00001111) << 4 | (src_buffer[src_i+19] & 0b11110000)>> 4;
        dst_buffer[dst_i+15] = (src_buffer[src_i+7] & 0b11110000) >> 4;

        dst_buffer[dst_i+16] = (src_buffer[src_i+8] & 0b00001111) << 4 | (src_buffer[src_i+20] & 0b00001111);
        dst_buffer[dst_i+17] = (src_buffer[src_i+8] & 0b11110000) >> 4;

        dst_buffer[dst_i+18] = (src_buffer[src_i+9] & 0b00001111) << 4 | (src_buffer[src_i+20] & 0b11110000)>> 4;
        dst_buffer[dst_i+19] = (src_buffer[src_i+9] & 0b11110000) >> 4;

        dst_buffer[dst_i+20] = (src_buffer[src_i+10] & 0b00001111) << 4 | (src_buffer[src_i+21] & 0b00001111);
        dst_buffer[dst_i+21] = (src_buffer[src_i+10] & 0b11110000) >> 4;

        dst_buffer[dst_i+22] = (src_buffer[src_i+11] & 0b00001111) << 4 | (src_buffer[src_i+21] & 0b11110000)>> 4;
        dst_buffer[dst_i+23] = (src_buffer[src_i+11] & 0b11110000) >> 4;

        dst_buffer[dst_i+24] = (src_buffer[src_i+12] & 0b00001111) << 4 | (src_buffer[src_i+22] & 0b00001111);
        dst_buffer[dst_i+25] = (src_buffer[src_i+12] & 0b11110000) >> 4;

        dst_buffer[dst_i+26] = (src_buffer[src_i+13] & 0b00001111) << 4 | (src_buffer[src_i+22] & 0b11110000)>> 4;
        dst_buffer[dst_i+27] = (src_buffer[src_i+13] & 0b11110000) >> 4;

        dst_buffer[dst_i+28] = (src_buffer[src_i+14] & 0b00001111) << 4 | (src_buffer[src_i+23] & 0b00001111);
        dst_buffer[dst_i+29] = (src_buffer[src_i+14] & 0b11110000) >> 4;

        dst_buffer[dst_i+30] = (src_buffer[src_i+15] & 0b00001111) << 4 | (src_buffer[src_i+23] & 0b11110000)>> 4;
        dst_buffer[dst_i+31] = (src_buffer[src_i+15] & 0b11110000) >> 4;
    }
}
