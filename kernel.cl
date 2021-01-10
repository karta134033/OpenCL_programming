__kernel void convolution(__global float *inputImage, __global float *outputImage, __global char *filter, int filter_width, int image_width, int image_height) {
    int halffilter_size = filter_width / 2;
    int row = get_global_id(1);
    int col = get_global_id(0);
    int sum = 0;
    char k_start = -halffilter_size + row >= 0 ? -halffilter_size : 0;
    char k_end = halffilter_size + row < image_height ? halffilter_size : halffilter_size + row - image_height - 1;
    char l_start = -halffilter_size + col >= 0 ? -halffilter_size : 0;
    char l_end = halffilter_size + col < image_width ? halffilter_size : halffilter_size + col - image_width - 1;
    for (int k = k_start; k <= k_end; ++k)
        for (int l = l_start; l <= l_end; ++l) 
            sum += inputImage[(row + k) * image_width + col + l] * filter[(k + halffilter_size) * filter_width + l + halffilter_size];

    outputImage[row * image_width + col] = sum;
}