#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h> 
#include "hostFE.h"
#include "helper.h"

void check_filter(float *filter, char *char_filter, int *filter_width) {
	int check_row = 0;
	int new_filter_width = *filter_width;
	int check_start = 0;
	int check_end = *filter_width - 1;
	bool check = true;
	while(check && check_start < check_end) {
		for (int i = 0; i < *filter_width && check; i++) if(filter[check_start * *filter_width + i] != 0) check = false;  // upper
		for (int i = 0; i < *filter_width && check; i++) if(filter[check_end * *filter_width + i] != 0) check = false;  // lower
		for (int i = 0; i < *filter_width && check; i++) if(filter[i * *filter_width + check_start] != 0) check = false;  // left
		for (int i = 0; i < *filter_width && check; i++) if(filter[i * *filter_width + check_end] != 0) check = false;  // right
		if (check) new_filter_width -= 2;
		check_start++;
		check_end--;
	}
	int char_filter_start = (*filter_width - new_filter_width) % 2 == 0 ? (*filter_width - new_filter_width) / 2 : 0;
	for (register int i = 0; i < new_filter_width; ++i)
		for (register int j = 0; j < new_filter_width; ++j)
			char_filter[i * new_filter_width + j] = filter[((char_filter_start + i) * *filter_width) + char_filter_start + j];

	*filter_width = new_filter_width;
	return;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
        float *inputImage, float *outputImage, cl_device_id *device,
        cl_context *context, cl_program *program) {

	int image_size = imageHeight * imageWidth;
	char *char_filter = (char *)malloc(filterWidth * filterWidth * sizeof(char));

	cl_command_queue cl_queue = clCreateCommandQueue(*context, *device, 0, NULL);
	cl_mem cl_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(float) * image_size, NULL, NULL);
	clEnqueueWriteBuffer(cl_queue, cl_inputImage, 0, 0, sizeof(float) * image_size, (void *)inputImage, 0, NULL, NULL);

	check_filter(filter, char_filter, &filterWidth);
	cl_mem cl_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(char) * filterWidth * filterWidth, NULL, NULL);
	clEnqueueWriteBuffer(cl_queue, cl_filter, 0, 0, sizeof(char) * filterWidth * filterWidth, (void *)char_filter, 0, NULL, NULL);
	cl_mem cl_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float) * image_size, NULL, NULL);
	cl_kernel convolution = clCreateKernel(*program, "convolution", NULL);

	size_t localws[2] = {8, 8};
	size_t globalws[2] = {imageWidth, imageHeight};

	clSetKernelArg(convolution, 1, sizeof(cl_mem), (void *)&cl_outputImage);
	clSetKernelArg(convolution, 3, sizeof(cl_int), (void *)&filterWidth);
	clSetKernelArg(convolution, 4, sizeof(cl_int), (void *)&imageWidth);
	clSetKernelArg(convolution, 5, sizeof(cl_int), (void *)&imageHeight);
	clSetKernelArg(convolution, 2, sizeof(cl_mem), (void *)&cl_filter);
	clSetKernelArg(convolution, 0, sizeof(cl_mem), (void *)&cl_inputImage);

	clEnqueueNDRangeKernel(cl_queue , convolution, 2, NULL, globalws, localws, 0, NULL, NULL);
	clEnqueueReadBuffer(cl_queue, cl_outputImage, 0, 0, sizeof(float) * image_size, (void *)outputImage, 0, NULL, NULL);
	
	free(char_filter);
}