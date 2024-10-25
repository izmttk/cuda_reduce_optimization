#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// 使用 shuffle 指令，即 __shfl_down_sync 原语
// shuffle 指令可以允许同一个 warp 内两个线程相互访问对方的寄存器
// 可以用来在同一个 warp 的不同线程间移动数据
// T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
// T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
// mask: 用于确定哪些线程可以访问，一般使用 __activemask()。
// var: 要移动的数据。
// delta: 移动的距离。
// T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);
// T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
// laneMask: 用于确定哪些线程可以访问，一般使用 1 << tid。
// srcLane: 源线程的 id。
__device__ void warp_reduce(volatile float *sdata, int tid, int n) {
    int laneid = tid % warpSize;
    float sum = sdata[tid] + sdata[tid + 32];
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    sdata[tid] = sum;
}

#define COARSE_FACTOR 2 // COARSE_FACTOR 只在从全局内存读取数据时有用
__global__ void reduce_sum_v9(float *input, float *output, int n) {
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

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride && i + stride < n) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 当 stride <= 32 时，只剩 32 个线程，直接在一个 warp 内部进行 reduce
    warp_reduce(sdata, tid, n);
    
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

    reduce_sum_v9<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input_d, output_d, n);

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