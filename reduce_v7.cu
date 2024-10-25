#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// thread coarsening
#define COARSE_FACTOR 4 // COARSE_FACTOR 只在从全局内存读取数据时有用
__global__ void reduce_sum_v7(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * COARSE_FACTOR) + threadIdx.x;

    float sum = 0;
#pragma unroll // 循环展开
    for (int tile = 0; tile < COARSE_FACTOR; tile++) {
        int idx = i + tile * blockDim.x;
        if (idx < n) {
            sum += input[idx];
        }
    }  
    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && i + stride < n) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

float *generate_data(int n) {
    srand(time(NULL)); // time.h 包含在了 cuda_runtime.h 中
    float *data = new float[n];
    for (int i = 0; i < n; i++) {
        data[i] = rand() % 10;
    }
    return data;
}

int main() {
    int n = 1 << 20;
    float *input_h = generate_data(n);
    float *output_h = new float;
    
    int blockSize = 1024;
    int numBlocks = (n + blockSize * COARSE_FACTOR - 1) / (blockSize * COARSE_FACTOR);

    float *input_d, *output_d;
    cudaMalloc(&input_d, n * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));

    cudaMemcpy(input_d, input_h, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_v7<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input_d, output_d, n);

    cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    
    // c++17 特性
    float sum = std::reduce(input_h, input_h + n);
    std::cout << "sum (cpu): " << sum << std::endl;
    std::cout << "sum (gpu): " << *output_h << std::endl;

    if (sum == *output_h) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    delete[] input_h;
    delete output_h;
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}