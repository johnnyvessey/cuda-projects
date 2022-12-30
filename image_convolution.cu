
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;

using std::vector;
using std::cout;


__global__ void Convolve(unsigned char* out, unsigned char *pixels, unsigned width, unsigned height, float *convolution, unsigned convWidth, unsigned convHeight) 
{
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int color = threadIdx.z;

    int col_offset = convWidth / 2;
    int row_offset = convHeight / 2;

    float sum = 0.0;

    for (int i = row - row_offset; i <= row + row_offset; i++)
    {
        if (i < 0 || i >= height)
            continue;
        for (int j = col - col_offset; j <= col + col_offset; j++)
        {
            if (j < 0 || j >= width)
                continue;

            int convRow = i - (row - row_offset);
            int convCol = j - (col - col_offset);
            int convIdx = convRow * convWidth + convCol;

            int pixelIdx = (i * width + j) * 4 + color;
            int pixelVal = pixels[pixelIdx];
            sum = sum + (convolution[convIdx] * pixelVal);

        }
    }
    int idx = (row * width + col) * 4 + color;

    if (color == 3)
    {
        out[idx] = 255;
    }
    else {
        out[idx] = sum;
    }

   

}

vector<unsigned char> ConvolveImage(vector<unsigned char>& pixels, vector<vector<float>> &convolution, unsigned width, unsigned height)
{
    unsigned char* input;
    unsigned char* out;

    int convSize = convolution.size() * convolution[0].size();
    unsigned convHeight = convolution.size();
    unsigned convWidth = convolution[0].size();
    float* conv = (float*)malloc(convSize * sizeof(float));

    float* cudaConv;

    for (int i = 0; i < convolution.size(); i++)
    {
        for (int j = 0; j < convolution[0].size(); j++)
        {
            conv[i * convWidth + j] = convolution[i][j];
        }
    }


    int pixelCount = pixels.size();

    int pixelsMemory = sizeof(unsigned char) * pixelCount;
    unsigned char* pixelPtr = (unsigned char*)malloc(pixelsMemory);

    for (int i = 0; i < pixelCount; i++)
    {
        pixelPtr[i] = pixels[i];
    }

    cudaMalloc((void**) &input, pixelsMemory);
    cudaMalloc((void**) &out, pixelsMemory);
    cudaMalloc((void**) &cudaConv, convSize * sizeof(float));

    cudaMemcpy(cudaConv, conv, convSize * sizeof(float), cudaMemcpyHostToDevice);
    free(conv);

    cudaMemcpy(input, pixelPtr, pixelsMemory, cudaMemcpyHostToDevice);
    free(pixelPtr);

    dim3 pixelGrid(width / 8, height / 8);
    dim3 subGrid(8, 8, 4);
    auto start = high_resolution_clock::now();
    Convolve << <pixelGrid, subGrid >> > (out, input, width, height, cudaConv, convWidth, convHeight);
    auto end = high_resolution_clock::now();

    std::cout << "Time: " << duration_cast<microseconds>(end - start).count() << "\n";
    unsigned char* outputPointer = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));
    cudaMemcpy(outputPointer, out, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(cudaConv);

    vector<unsigned char> outputPixels;
    outputPixels.reserve(pixelCount);

    for (int i = 0; i < pixelCount; i++)
    {
        outputPixels.push_back(outputPointer[i]);
    }

    free(outputPointer);
    cudaFree(out);

    return outputPixels;
}
int main(void) {
    vector<unsigned char> pixels;
    unsigned int width = 1024;
    unsigned int height = 1024;
    unsigned error = lodepng::decode(pixels, width, height, "test_image.png");
    
    vector<vector<float>> convolution = { {0, -1, 0}, {-1, 5 , -1}, {0, -1 , 0} };
    vector<unsigned char> newImage = ConvolveImage(pixels, convolution, width, height);


    lodepng::encode("conv_image.png", newImage, width, height);
    std::cout << "Completed convolution\n";

}
