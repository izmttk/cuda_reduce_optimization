#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// 使用 shared memory
// 同时减少空闲线程数
__global__ void reduce_sum_v6(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (i + blockDim.x < n) {
        sdata[tid] = input[i] + input[i + blockDim.x];
    } else if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0;
    }
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
    int numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);

    float *input_d, *output_d;
    cudaMalloc(&input_d, n * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));

    cudaMemcpy(input_d, input_h, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_v6<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input_d, output_d, n);

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