/*
    This is a sample application for the Jetson kit (XEC-NX-3P-X2G3-ORIN-16GB-KIT) . It demostrates basic image processing on GPU, circumventing xiAPI processing.
    As xiAPI only runs on CPU, and as of 2025 does not have SIMD acceleration on ARM platfroms, this provides a significant speedup.
    This also serves as a basic template for further development of your computer vision application on Nvidia Jetson.
    
    Sample name:
        jetson_sample

    Description:
        Open camera, set arbitrary camera parameters, in a loop capture image into CUDA unified memory, execute image depacking 
        with custom CUDA kernel, do debayering with vision library of your choosing (OpenCV with CUDA or Nvidia performance primitives),
        do image normalization and save to disk with OpenCV and render onto OpenCV window with OpenGL support.
        Print statistics of image processing operations.
    
    Workflow:
        1. Open camera
        2. Set camera parameters
        3. Allocate CUDA unified memory buffers
        4. Set variables for demosaicing
        5. Create rendering window with OpenGL backend
        6. Start acquisition 
        7. Receive image from the camera
        8. Depack image with CUDA kernel
        9. Demosaic image with selected vision library
        10.Normalize image
        11.Save image to .tif
        12.Render image
        13.Print statistics
        14.Cleanup and close program

*/
#include <iostream>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <thread>

#include "time_measurement.h"

#include <m3api/xiApi.h>

#include <opencv2/core/cvstd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs.hpp>

#include <npp.h>
#include <nppdefs.h>
#include <cuda_runtime.h>

#include "depacking.h"

using namespace std;

#define NUM_CYCLES 20

//#define OPENCV_DEMOSAICING


#define SLEEP(n) std::this_thread::sleep_for(std::chrono::milliseconds(n))

#define ABORT_PROGRAM  cout << "Aborting program due to errors!" << endl; exit(-1); //note that these are two instructions
                                                                                    //use with curly braces

#define CHECK_XI_RET_EXCEPT(str) if (ret != XI_OK) throw str;
#define CHECK_XI_RET(str) if (ret!=XI_OK) {cout<<"XIAPI error at: " << str << endl; ABORT_PROGRAM }
#define CHECK_CUDA_RET(str) if (ret!=cudaSuccess) {cout<<"CUDA error at: " << str << endl; ABORT_PROGRAM }
#define CHECK_NPP_RET(str) if (ret!=NPP_NO_ERROR) {cout<<"NPP error at: " << str << endl; ABORT_PROGRAM }



/**
 * @brief Opens camera, sets parameters, retrieves cfa and transport format
 * @param xiH camera handle
 * @return -1 at failure, 0 at success
 */
int OpenCamera(HANDLE & xiH, int & cfa_type, int & transport_format)
{
    int ret = XI_OK;

    try 
    {
        //Open device, opening first
        ret = xiOpenDevice(0, &xiH);
        CHECK_XI_RET_EXCEPT("Camera could not be opened");

        //Setting buffer policy, for CUDA with USB camera using SAFE with user allocated buffer
        ret = xiSetParamInt(xiH, XI_PRM_BUFFER_POLICY, XI_BP_SAFE);
        CHECK_XI_RET_EXCEPT("Setting buffer policy failed");

        /*
        Format: XI_RAW16 Parameters controlled automatically:
        XI_PRM_SENSOR_DATA_BIT_DEPTH = maximum
        XI_PRM_OUTPUT_DATA_BIT_DEPTH = SENSOR_DATA_BIT_DEPTH
        XI_PRM_OUTPUT_DATA_PACKING = ON 
        Image Processing: disabled
        */        

        ret = xiSetParamInt(xiH, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW16);
        CHECK_XI_RET_EXCEPT("Setting image data format failed");

        
        //Setting format as transport, no image processing by xiAPI
        ret = xiSetParamInt(xiH, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
        CHECK_XI_RET_EXCEPT("Setting image data format failed");


        //Verify that packing is set on
        int packing = XI_OFF;
        ret = xiGetParamInt(xiH, XI_PRM_OUTPUT_DATA_PACKING, &packing);
        CHECK_XI_RET_EXCEPT("Retrieving data packing failed");
    
        if (packing != XI_ON)
        {
            cout << "Packing was not set on automatically. Setting on manually." << endl;
            ret = xiSetParamInt(xiH, XI_PRM_OUTPUT_DATA_PACKING, XI_ON);
            CHECK_XI_RET_EXCEPT("Setting packing failed");
        }

        ret = xiGetParamInt(xiH, XI_PRM_TRANSPORT_PIXEL_FORMAT, &transport_format);
        CHECK_XI_RET_EXCEPT("Retrieving packing type failed");

        cout << "TRANSPORT FORMAT: " << std::hex << transport_format << endl;

        //Setting exposure to arbitrary value
        ret = xiSetParamInt(xiH, XI_PRM_EXPOSURE, 1600);
        CHECK_XI_RET_EXCEPT("Setting exposure failed");

        //Setting gain to arbitrary value
        ret = xiSetParamFloat(xiH, XI_PRM_GAIN, 5);
        CHECK_XI_RET_EXCEPT("Setting gain failed");

      
        //check whether the macros are correct   
        ret = xiGetParamInt(xiH, XI_PRM_COLOR_FILTER_ARRAY, &cfa_type);
        CHECK_XI_RET_EXCEPT("Getting cfa failed");

        //Checking  CFA types
        if (cfa_type == XI_CFA_NONE)
            throw "No CFA detected! Plese connect a color camera.";
        else if (!(cfa_type == XI_CFA_BAYER_RGGB || cfa_type == XI_CFA_BAYER_BGGR || cfa_type == XI_CFA_BAYER_GRBG || cfa_type == XI_CFA_BAYER_GBRG))
            throw "Unsupported CFA type! This sample supports only Bayer types.";
   
    }
    catch(const char* msg)
    {
        cerr << "Error in opening camera: ";
        cerr << msg << endl;
        return -1;
    }

    return 0;
}

/**
 * @brief Converts XI CFA variable to NPPI CFA 
 * @param cfa_type XI_COLOR_FILTER_ARRAY from camera
 */
NppiBayerGridPosition XiCFAToNppCFA(int cfa_type)
{
    //The inversion here is necessary, because cameras return BGR
    //and NPPI handles only RGB
    switch (cfa_type)
    {
        case XI_CFA_BAYER_RGGB: return NPPI_BAYER_BGGR; 
        case XI_CFA_BAYER_BGGR: return NPPI_BAYER_RGGB; 
        case XI_CFA_BAYER_GRBG: return NPPI_BAYER_GRBG;
        case XI_CFA_BAYER_GBRG: return NPPI_BAYER_GBRG;
    }
    return (NppiBayerGridPosition)0;
}

/**
 * @brief Converts XI CFA variable to OpenCV CFA 
 * @param cfa_type XI_COLOR_FILTER_ARRAY from camera
 */
cv::ColorConversionCodes XiCFAToOCVCFA(int cfa_type)
{
    switch (cfa_type)
    {
        case XI_CFA_BAYER_RGGB: return cv::COLOR_BayerRG2BGR; 
        case XI_CFA_BAYER_BGGR: return cv::COLOR_BayerBG2BGR; 
        case XI_CFA_BAYER_GRBG: return cv::COLOR_BayerGR2BGR;
        case XI_CFA_BAYER_GBRG: return cv::COLOR_BayerGB2BGR;
    }
    return (cv::ColorConversionCodes)0;
}


int main()
{
    int ret = XI_OK;

    //Open Camera
    HANDLE xiH = NULL;
    int cfa_type = XI_CFA_NONE; //initialization value
    int transport_format = XI_GenTL_Image_Format_Mono8; //initialization value
    
    ret = OpenCamera(xiH, cfa_type, transport_format);
    CHECK_XI_RET("Opening camera")
    
    cout << "Opened camera" << endl;
  
    //Retrieve image dimensions
    int height = 0;
    ret = xiGetParamInt(xiH, XI_PRM_HEIGHT, &height);
    CHECK_XI_RET("Retrieving height")

    int width = 0;
    ret = xiGetParamInt(xiH, XI_PRM_WIDTH, &width);
    CHECK_XI_RET("Retrieving width")
      
    //Retrieve necessary buffer size
    int raw_buffer_length = 0;
    ret = xiGetParamInt(xiH, XI_PRM_IMAGE_PAYLOAD_SIZE, &raw_buffer_length);
    CHECK_XI_RET("Retrieving buffer size")
    
    //Display parameters
    cout << "Image height: " << std::dec << height << endl;
    cout << "Image width: " << std::dec << width << endl;
    cout << "Buffer size: " << std::dec << raw_buffer_length << endl;

    //Allocating buffer for XI_IMG, using BP_SAFE with user allocated buffer
    //See https://www.ximea.com/support/gfiles/buffer_policy_in_xiApi.png
    XI_IMG cap_image;

    //This initialization of the XI_IMG structure is required
    memset(&cap_image, 0, sizeof(XI_IMG)); 
    cap_image.size = sizeof(XI_IMG);

    cap_image.bp_size=raw_buffer_length;

    ret = cudaMallocManaged (&cap_image.bp, raw_buffer_length, cudaMemAttachGlobal);
    CHECK_CUDA_RET("Allocating unified memory to cap_image.bp variable")

    //Allocate buffer for depacked frame
    uint8_t* depacked_buffer;
    ret = cudaMallocManaged (&depacked_buffer, width*height*2, cudaMemAttachGlobal);
    CHECK_CUDA_RET("Allocating unified memory to depacked_buffer variable")

    //Allocate buffer for demosaiced frame
    uint8_t* color_buffer;
    ret = cudaMallocManaged (&color_buffer, width*height*2*3, cudaMemAttachGlobal);
    CHECK_CUDA_RET("Allocating unified memory to color_buffer variable")


    //Parse CFA
    #ifdef OPENCV_DEMOSAICING
    cv::ColorConversionCodes OCV_cfa = XiCFAToOCVCFA(cfa_type);

    //OCV vars
    cv::cuda::GpuMat gpu_mat_out(height, width, CV_16UC3, color_buffer);
    cv::cuda::GpuMat gpu_mat_in(height, width, CV_16UC1, depacked_buffer);

    #else

    NppiBayerGridPosition NPP_cfa = XiCFAToNppCFA(cfa_type);
    
    //NPPI vars
    NppiSize img_size;
    img_size.height = height;
    img_size.width = width;

    NppiRect img_roi{0,0,width,height}; // no ROI, ROI is full image

    #endif

    //Create OpenCV window
    char window_name[] = "Opencv window";
    char window_title[] = "Live image";

    cv::namedWindow(window_name,cv::WINDOW_OPENGL);
    cv::setWindowTitle(window_name,window_title);
    cv::resizeWindow(window_name,width,height);//This has to be here otherwise the window will not work. Reason unknown.

    //Start acquisition
    ret = xiStartAcquisition(xiH);
    CHECK_XI_RET("Starting acquisition")
    
    //Time statistics variables
    unsigned long time_0=0,time_1=0,time_2=0,time_3=0, time_4=0, time_5=0, time_6=0, time_7=0;     
    unsigned long depacking_time=0, demosaicing_time=0, saving_time=0, rendering_time=0;

    for (int it=0; it<NUM_CYCLES; it++)
    {
        //Retrieve image
        ret = xiGetImage(xiH, 1000, &cap_image);
        CHECK_XI_RET("Retrieving captured image")

        cout << "Processing frame " << it << endl;
        
        //Depack image
        time_0 = GetCurTimeUs();
        DepackBuffer((uint8_t*)cap_image.bp,depacked_buffer,transport_format,width,height);
        time_1 = GetCurTimeUs();

        cout << "NPPI demosaicing!" << endl;
        
        time_2 = GetCurTimeUs();

        #ifdef OPENCV_DEMOSAICING
        
        cv::cuda::demosaicing(gpu_mat_in, gpu_mat_out, OCV_cfa,3);
                
        #else
        //Demosaicing
        ret = nppiCFAToRGB_16u_C1C3R((Npp16u*)depacked_buffer,
                                                width*2, //stride is in bytes
                                                img_size,
                                                img_roi,
                                                (Npp16u*)color_buffer,
                                                width*2*3,
                                                NPP_cfa,
                                                NPPI_INTER_UNDEFINED);
        CHECK_NPP_RET("Demosaicing") 
        
        #endif
        
        ret = cudaDeviceSynchronize();
        if (ret!= cudaSuccess)
        { 
            cerr << "Unknown failure during debayering!" << endl;
        }
        
        time_3 = GetCurTimeUs();
        /*
          Note: The demosaiced image is going to have a green tint. 
          This can be rectified by applying white balance here.
        */

        //Encapsule demosaiced image in cv::Mat for openCV processing
        cv::Mat export_image (height,width,CV_16UC3); 
        export_image.data = (uchar*) color_buffer;

        //Image normalization
        cv::normalize(export_image,export_image,0,65535,cv::NORM_MINMAX);

        char filename[] = "saved_image_00.tif" ;
        sprintf(filename,"saved_image_%02d.tif",it);

        cout << "Saving to .tif!" << endl;

        //Saving to disk
        time_4 = GetCurTimeUs();
        cv::imwrite(filename, export_image);
        time_5 = GetCurTimeUs();

        //Rendering image
        cout << "Rendering!" << endl;
        
        time_6 = GetCurTimeUs();
        cv::imshow(window_name,export_image);
        time_7 = GetCurTimeUs();

        int keycode = cv::waitKey(1) & 0xff ; 
        //Not doing any key press processing, but the option is here
        
        //In case you want to introduce delay between cycles, uncomment the following line
        //SLEEP(1);

        depacking_time += time_1-time_0;
        demosaicing_time += time_3-time_2;
        saving_time += time_5-time_4;
        rendering_time += time_7-time_6;
    }

    //Print statistics
    cout << "\nStatistics:" << endl;
    cout << "Average depacking time: " << (depacking_time/NUM_CYCLES)/1000.0 << " ms" << endl;
    cout << "Average demosaicing time: " << (demosaicing_time/NUM_CYCLES)/1000.0 << " ms" << endl;
    cout << "Average saving time: " << (saving_time/NUM_CYCLES)/1000.0 << " ms" << endl;
    cout << "Average Rendering time: " << (rendering_time/NUM_CYCLES)/1000.0 << " ms" << endl;
    cout << "\n" << endl;

    //Stop acquisition
    ret = xiStopAcquisition(xiH);
    CHECK_XI_RET("Stopping acquisition")

    //Destroy OpenCV window
    cv::destroyAllWindows();

    //Close camera
    ret = xiCloseDevice(xiH);
    CHECK_XI_RET("Closing camera")    

    ret = cudaFree(cap_image.bp);
    if (ret != cudaSuccess)
    {
        cerr << "Could not deallocate cap_image.bp variable! Clean memory leaks manually." << endl;
    }

    ret = cudaFree(depacked_buffer);
    if (ret != cudaSuccess)
    {
        cerr << "Could not deallocate depacked_buffer variable! Clean memory leaks manually." << endl;
    }

    ret = cudaFree(color_buffer);
    if (ret != cudaSuccess)
    {
        cerr << "Could not deallocate depacked_buffer variable! Clean memory leaks manually." << endl;
    }

    return 0;
}

