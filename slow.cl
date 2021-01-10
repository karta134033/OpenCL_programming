__kernel void convolution(
    __global float *filter, __global float *inputImage, __global float *outputImage, int filterWidth, int imageHeight, int imageWidth, int halffilterSize) {
        
    int filter_size = filterWidth * filterWidth;
    int row = get_global_id(1);
    int col = get_global_id(0);
    __local float shared_filter[49];
    for (int i = 0; i < filter_size; i++)
        shared_filter[i] = filter[i];

    float sum = 0.f;
    int k_start = -halffilterSize + row >= 0 ? -halffilterSize : 0;
    int k_end = halffilterSize + row < imageHeight ? halffilterSize : halffilterSize + row - imageHeight - 1;
    int l_start = -halffilterSize + col >= 0 ? -halffilterSize : 0;
    int l_end = halffilterSize + col < imageWidth ? halffilterSize : halffilterSize + col - imageWidth - 1;
    
    for (int k = k_start; k <= k_end; ++k)
        for (int l = l_start; l <= l_end; ++l) 
            sum += inputImage[(row + k) * imageWidth + col + l] * shared_filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
        
    outputImage[row * imageWidth + col] = sum;
}