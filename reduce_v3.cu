#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// Interleaved Addressing 交错寻址，即把具有相同条件的线程相邻放置（或者说放在同一个 warp 中）
// 优化 warp divergence
__global__ void reduce_sum_v3(float *input, float *output, int n) {
    // 在每个 block 内进行独立的 reduce
    int size = blockDim.x * 2;
    int offset = blockIdx.x * size;
    int tid = threadIdx.x;
    for (int stride = 1; stride < size; stride *= 2) {
        // 每个线程负责一次 reduce
        int i = tid * 2 * stride; // 被加和元素的块内位置
        if (i < size && offset + i + stride < n) {
            input[offset + i] += input[offset + i + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, input[offset]);
    }
}

float *generate_data(int n) {
    srand(time(NULL));
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
    // numBlocks = ceil((n / 2) / blockSize) = ceil(n / (blockSize * 2))
    int numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);

    float *input_d, *output_d;
    cudaMalloc(&input_d, n * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));

    cudaMemcpy(input_d, input_h, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_v3<<<numBlocks, blockSize>>>(input_d, output_d, n);

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