---
tags: 平行程式設計
---

# OpenCL Programming

## Q1 
>(5 points): Explain your implementation. How do you optimize the performance of convolution?

1. **將傳輸資料量縮小**
    原本傳入的圖片為"float"型態，佔用空間為4bytes。
    圖像處理中常見的型態為"uchar"，一般是指unsigned char，佔用空間為1bytes。
    從助教的"readImage()"function也可以得知讀入的圖片是以**uchar做儲存再轉為float**，因此確實可以將"**輸入**"圖片轉為uchar的型態。
    雖然"**輸出**"圖片的數值最後也是以uchar來表示，但做"**Diff ratio"的比較時是以float做比較**，也因此沒辦法直接將格式由floar轉為uchar(直接轉的話雖然圖片是正確的，但Diff ratio會無法通過)，因此輸出圖片的型態不變。
    
    ```c
    float *inputImage 改為 unsigned char *inputImage
    ```
    
2. **修改for迴圈寫法**
    原先的for迴圈寫法會不斷的做if判斷，檢查計算的內容有沒有超出圖片邊界，速度上會慢上許多。
    原做法:
    ```c
    for (k = -halffilterSize; k <= halffilterSize; k++){
        for (l = -halffilterSize; l <= halffilterSize; l++){
            if (i + k >= 0 && i + k < imageHeight &&   // <--持續判斷
                j + l >= 0 && j + l < imageWidth){
                sum += inputImage[(i + k) * imageWidth + j + l] *
                    filter[(k + halffilterSize) * filterWidth +l + halffilterSize];
            }
        }
    }
    ```
    原本的作法可以改良為將k, l的開始與結束位置在之前就計算好再帶入for的條件中。
    改良後:
    ```c
    int k_start = -halffilterSize + row >= 0 ? -halffilterSize : 0;  // 計算起點
    int k_end = halffilterSize + row < imageHeight ? halffilterSize :  // 計算終點
        halffilterSize + row - imageHeight - 1;
    int l_start = -halffilterSize + col >= 0 ? -halffilterSize : 0;
    int l_end = halffilterSize + col < imageWidth ? halffilterSize : 
        halffilterSize + col - imageWidth - 1;
    
    for (int k = k_start; k <= k_end; ++k)
        for (int l = l_start; l <= l_end; ++l) 
            sum += inputImage[(row + k) * imageWidth + col + l] * 
                shared_filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
    
    ```

3. **修改filter的作用域** 
    每個thread都會使用到相同的filter，頻繁的讀取資料會拖慢kernel速度，因此將filter拷貝到速度較快的"__local"中。   
    除此之外，有try過將圖片的內容依照group的id拷貝對應的區塊到"__local"中，但效能上並未有顯著提升。可能是切的方式還不夠完整，因此沒放入報告中。
    
4. **修剪filter的大小** (CUDA只實作到此版本)
    <font color="red">此方式是"動態"修改的方式，若不符合修改原則則會以原filter做為輸入</font>
    檢查filter2.csv與filter3.csv
    ```t
    // filter2
    0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 
    0 0 1 0 1 0 0 
    0 0 2 0 2 0 0 
    0 0 1 0 1 0 0 
    0 0 0 0 0 0 0 
    0 0 0 0 0 0 0
    
    // filter3
    0 0 0 0 0
    0 1 0 1 0
    0 1 1 1 0
    0 1 1 1 0
    0 0 0 0 0
    ```
    可以發現filter的外圈都是"0"，等於說會影響卷積結果的部分都只有中間 3 * 3的內容，因此可以將這兩個filter分別修剪為
    
    ```t
    // filter2
    1 0 1
    2 0 2
    1 0 1 
    
    // filter3
    1 0 1 
    1 1 1 
    1 1 1 
    ```

5. **unroll**
    從上一點可得知filter**如果**最後是3 * 3的結果，for迴圈也會固定是3 * 3的內容，根據此想法可以將原本卷積的部分改寫為
    ```c
    for (char k = k_start; k <= k_end; ++k) {
        char l = l_start;
        sum += inputImage[(row + k) * IMAGE_W + col + l] * 
            shared_filter[(k + HALF_FILTER_W) * FILTER_W + l++ + HALF_FILTER_W];
        sum += inputImage[(row + k) * IMAGE_W + col + l] * 
            shared_filter[(k + HALF_FILTER_W) * FILTER_W + l++ + HALF_FILTER_W];
        sum += inputImage[(row + k) * IMAGE_W + col + l] * 
            shared_filter[(k + HALF_FILTER_W) * FILTER_W + l + HALF_FILTER_W];
    }
    ```
    不過，filter有可能無法修剪為3 * 3，如果不是3 * 3的內容，則會照著原本的for loop方式。

6. **檢查filter是否是預先設定好的** <font color="red">(為避免爭議，最後上傳的版本不含此功能)</font>
    實做Open CL版本的內容時意外發現，若能在__kernel定義預先要使用的filter，再根據傳入的filter判斷，若傳入的filter與預先定義好的filter完全相同，則直接使用預先定義好的filter做卷積的方法在速度上會快上許多。
    <font color="red">若傳入的filter與預先訂義好的filter內容不同，卷積時的filter則會以原先的filter為主。</font>
    不過此優化方式在cuda卻沒有明顯差異!所以在cuda的實作上並未加入此功能。
    
7. **將資料傳輸方式改為異步**
    
----
報告闡述的版本可能還不夠通用，所以還有實作其他的版本。
* 0.4940
![](https://i.imgur.com/fircnWc.png =450x300)
* 0.6047
![](https://i.imgur.com/UwT3Q6h.png =450x300)
* 0.7160 
**此版本為最後上傳的版本**，不過在工作站上的執行時間滿不穩定的，最終結果可能不同。
![](https://i.imgur.com/2ZKn2Ve.png =450x300)
* 0.8737
![](https://i.imgur.com/VaraiOP.png =450x300)
* 0.9363
![](https://i.imgur.com/TBeWiLO.png =450x300)
* 0.9783
![](https://i.imgur.com/hb8UTWB.png =450x300)
* 1.1607
![](https://i.imgur.com/B1vMeQd.png =450x300)
* 1.2667
![](https://i.imgur.com/pYmNvD7.png =450x300)
* 1.4953
![](https://i.imgur.com/SSiZktm.png =450x300)
* 1.5847
![](https://i.imgur.com/WzTD7Da.png =450x300)
* 2.1620
![](https://i.imgur.com/a8jb6it.png =450x300)


以上的實作在優化上會有些許不同，但都會符合:
* 可測試不同的filter，不限制在3個測試的filter
* filter大小不限
* 圖片大小不限
* 沒有預設輸出值
* 計算內容皆會初始化

若有問題會再提供助教其他的版本。

還有一個0.0000違規的版本，放著做個紀念。
![](https://i.imgur.com/Q0tI6jY.png)
測試的Script還因此壞掉了:zany_face:


## Q2 
> (10 points): Rewrite the program using CUDA. (1) Explain your CUDA implementation, (2) plot a chart to show the performance difference between using OpenCL and CUDA, and (3) explain the result.

* (1)
    CUDA的實作方式與Open CL的版本差異不大，只是程式環境不大一樣。
    HW5的作業已經有給CUDA程式編譯用的makefile，為了節省時間決定將CUDA的版本實作在HW5的環境上，只需要在main做點更動並在kernel.cu實作出程式內容就完成了。
    
    實作方法:
    * 將傳輸資料量縮小(改變資料型態)
    * 修改for迴圈寫法
    * 修剪filter的大小<font color="red">(動態)</font>

    有些優化方式在CUDA沒有明顯的效益，所以沒有把它實作上去，在做性能評估時，Open CL的版本會下修為CUDA實作的方法，也就是說上傳e3的Open CL版本不會是以下比較性能時的版本。
    
    ```cpp
    #include <cuda.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define BLOCK_SIZE 8

    __global__ static void convolution(unsigned char *inputImage, unsigned short *outputImage, char *filter, int filter_width, int image_width, int image_height) {
        int halffilter_size = filter_width / 2;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
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

    // Host front-end function that allocates the memory and launches the GPU kernel
    void hostFE (int filterWidth, float * filter, int imageHeight, int imageWidth, float * inputImage, float * outputImage) {
        int image_size = imageHeight * imageWidth;
        char *char_filter = (char *)malloc(filterWidth * filterWidth * sizeof(char));
        unsigned char *uchar_input = (unsigned char *)malloc(image_size * sizeof(unsigned char));
        unsigned short *short_output = (unsigned short *)malloc(image_size * sizeof(unsigned short));
        for (register int i = 0; i < image_size; ++i) uchar_input[i] = inputImage[i];
        check_filter(filter, char_filter, &filterWidth);

        char *device_filter;
        unsigned char *device_input;
        unsigned short *device_output;
        cudaMalloc(&device_filter, filterWidth * filterWidth * sizeof(char));
        cudaMalloc(&device_input, image_size * sizeof(unsigned char));
        cudaMalloc(&device_output, image_size * sizeof(unsigned short));
        cudaMemcpy(device_filter, char_filter, filterWidth * filterWidth * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input, uchar_input, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

        static int x_blocks = imageWidth / BLOCK_SIZE;
        static int y_blocks = imageHeight / BLOCK_SIZE;
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 num_block(x_blocks, y_blocks);
        convolution<<<num_block, block_size>>>(device_input, device_output, device_filter, filterWidth, imageWidth, imageHeight);
        cudaMemcpy(short_output, device_output, image_size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        for (register int i = 0; i < image_size; ++i) outputImage[i] = (float)short_output[i];

        cudaFree(device_filter);
        cudaFree(device_input);
        cudaFree(device_output);

        free(char_filter);
        free(uchar_input);
        free(short_output);
    }
    ```
* (2)
    帶入的filter剛好都可以修剪為3*3的大小，所以效能評估上只做一個執行指令的比較。
    執行指令: ./conv
    <font color="red">此Open CL版本是比照CUDA的優化方式另外寫的，上傳的般本會與此不同。</font>
    ![](https://i.imgur.com/6lPqKvU.png)

* (3)
    原本預期CUDA會比較快一點，因為GPU是nvidia的，下意識會認為相容性、效能、優化等...會使結果"明顯"優於Open CL，但以實驗結果來看差異並不大。
    
    上網查了一下資料發現[這篇論文](https://arxiv.org/vc/arxiv/papers/1005/1005.2581v1.pdf)有對兩個架構的效能差異做分析。
    以下節錄幾個重點:
    > We tested CUDA and OpenCL versions of our application on an **NVIDIA** GeForce GTX260.

    > **OpenCL kernel can be compiled at runtime**, which would add to an OpenCL’s running time. On the other hand, this just-in-time compile **may allow the compiler to generate code that makes better use of the target GPU**. 
    
    > **CUDA, on the other hand, is developed by the same company that develops the hardware on which it executes, so one may expect it to better match thecomputing characteristics of the GPU, offering more access to features and better performance.** Considering these factors, it is of interest to compare OpenCL’s performance to that of CUDA in a real-world application.
    
    > In this paper we use a **computationally-intensive** scientific application to provide a performance comparison of CUDA and OpenCL on an NVIDIA GPU. 
    
    >For this paper we **optimized the kernel’s memory access patterns**. We then ported the CUDA kernel to OpenCL, a process which, with NVIDIA development tools, required minimal code changes in the kernel itself, as explained below. Other related code, for example to detect and setup the GPU or to copy data to and from the GPU, needed to be re-written for OpenCL.

    ![](https://i.imgur.com/ouj0Q47.png)
    ![](https://i.imgur.com/vDvjIlw.png)
    ![](https://i.imgur.com/ThqdoUg.png)


    由結果可以發現，在資料傳輸與計算方面CUDA皆是優於Open CL，但這個實驗是建構在**computationally-intensive**的實驗目標且對實作上有做最佳化處理。
    回顧到之前的卷積作法只是以3*3的filter去做圖像處理，計算量是相對少的，**儘管資料傳輸方面CUDA是優於Open CL，但在計算方面較難展現出明顯的差異，可能導致時間上與Open CL的做法差異不大**。此外，時間的差異也有可能是因為沒有最佳化CUDA的寫法所致。
    
    除了實驗結果外，在實作方面，以上手難度而言我認為Open CL > CUDA，主要分為兩項:
    1. Open CL的前置作業比較多(由作業中helper.c的initCL()可看出)
    2. 網路資源較少(畢竟在效能上輸CUDA，顯卡的主流又是以nvidia為主)

    未來若需要做GPU的計算我會以CUDA作為優先考量。
        
    