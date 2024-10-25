#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// 循环展开最后一个 warp 的 reduce
// 一个 warp level kernel，外层包一层使其成为一个 block level kernel，是一种很规范的 kernel 写法，方便规避 bank conflict。
__device__ void warp_reduce(volatile float *sdata, int tid, int n) {
    // 原本写法： sdata[tid] += sdata[tid + 32]; ...
    // 在 CC 7.0 （Volta 架构）之前，同一个 wrap 内所有线程共享一个 PC 指针，指令的执行顺序是严格同步的。
    // 在 CC 7.0 及之后，一个 wrap 内各线程拥有自己的 PC 指针，需要使用 __syncwarp() 来保证同步。
    // 我们要保证所有线程读完之后再写，直接 sdata[tid] += sdata[tid + 32]; __syncwarp(); 是错误的。
    // 因此需要将赋值语句拆分为读写两部分，中间插入同步原语。顺序：read shmem / sync / write shmem
    // float sum = sdata[tid];
    // sum += sdata[tid + 32]; __syncwarp(); sdata[tid] = sum;
    // sum += sdata[tid + 16]; __syncwarp(); sdata[tid] = sum;
    // sum += sdata[tid +  8]; __syncwarp(); sdata[tid] = sum;
    // sum += sdata[tid +  4]; __syncwarp(); sdata[tid] = sum;
    // sum += sdata[tid +  2]; __syncwarp(); sdata[tid] = sum;
    // sum += sdata[tid +  1]; __syncwarp(); sdata[tid] = sum;


    // 想法：判断了被激活的线程，就能避免读写共享内存的竞态问题，上述错误写法应该能够正确运行。
    // if (tid < 32 && tid + 32 < n) sdata[tid] += sdata[tid + 32]; __syncwarp();
    // if (tid < 16 && tid + 16 < n) sdata[tid] += sdata[tid + 16]; __syncwarp();
    // if (tid <  8 && tid +  8 < n) sdata[tid] += sdata[tid +  8]; __syncwarp();
    // if (tid <  4 && tid +  4 < n) sdata[tid] += sdata[tid +  4]; __syncwarp();
    // if (tid <  2 && tid +  2 < n) sdata[tid] += sdata[tid +  2]; __syncwarp();
    // if (tid <  1 && tid +  1 < n) sdata[tid] += sdata[tid +  1]; __syncwarp();

    // 继续利用 #pragma unroll 简化代码
#pragma unroll
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < n) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncwarp();
    }
}

#define COARSE_FACTOR 2 // COARSE_FACTOR 只在从全局内存读取数据时有用
__global__ void reduce_sum_v8(float *input, float *output, int n) {
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

    reduce_sum_v8<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input_d, output_d, n);

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