#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
     
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filter_size = filterWidth * filterWidth;
    int image_size = imageHeight * imageWidth;
    int halffilterSize = filterWidth / 2;
    int filter_W = filterWidth;
    int image_H = imageHeight;
    int image_W = imageWidth;

    cl_int status;
    cl_kernel convolution = clCreateKernel(*program, "convolution", &status);
    cl_command_queue cl_queue = clCreateCommandQueue(*context, *device, 0, &status);
    cl_mem cl_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(float) * filter_size, NULL, &status);
    cl_mem cl_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(float) * image_size, NULL, &status);
    cl_mem cl_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float) * image_size, NULL, &status);
    clEnqueueWriteBuffer(cl_queue, cl_filter, CL_TRUE, 0, sizeof(float) * filter_size, (void *)filter, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_queue, cl_inputImage, CL_TRUE, 0, sizeof(float) * image_size, (void *)inputImage, 0, NULL, NULL);

    clSetKernelArg(convolution, 0, sizeof(cl_mem), (void *)&cl_filter);
    clSetKernelArg(convolution, 1, sizeof(cl_mem), (void *)&cl_inputImage);
    clSetKernelArg(convolution, 2, sizeof(cl_mem), (void *)&cl_outputImage);
    clSetKernelArg(convolution, 3, sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(convolution, 4, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(convolution, 5, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(convolution, 6, sizeof(cl_int), (void *)&halffilterSize);

    size_t localws[2] = {20, 20};
    size_t globalws[2] = {imageWidth, imageHeight};

    cl_int ret = clEnqueueNDRangeKernel(cl_queue , convolution, 2, NULL, globalws, localws, 0, NULL, NULL);
    clEnqueueReadBuffer(cl_queue, cl_outputImage, CL_TRUE, 0, sizeof(float) * image_size, (void *)outputImage, 0, NULL, NULL);
    
}