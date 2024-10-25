#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>


// pytorch 的 block reduce 写法
#define COARSE_FACTOR 2 // COARSE_FACTOR 只在从全局内存读取数据时有用
#define WARP_SIZE 32
#define BLOCK_SIZE 1024
constexpr int threadsPerBlock = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

// 一个 warp 有 32 个线程
__device__ float warp_reduce(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 一个 block 最多 1024 个线程，即 32 个 warp
__device__ float block_reduce(float val, float* sdata) {
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    val = warp_reduce(val);
    // __syncthreads();
    // 如果要连续调用 block_reduce，需要在此处或者两个函数调用之间同步
    // 否则如果 sdata 相同，会发生读写冲突，举例如下：
    // sum1 = BlockReduceSum(val1, shared);
    // sum2 = BlockReduceSum(val2, shared);
    // 1、假设有block中有2个warp，在第二次__syncthreads()之后，warp1执行较快结束进入第二个BlockReduceSum中
    // 2、此时 warp0 正准备读取shared[1]，但warp1先在第二个BlockReduceSum中写入shared[1]，这就出现了读写冲突
    if (lane == 0) {
        sdata[warp] = val; // 写共享内存
    }
    // 同样在共享内存读写之间需要同步
    // 否则第1个 warp 在写完共享内存后开始读取了，其他 warp 还没写完
    __syncthreads();
    val = (tid < threadsPerBlock) ? sdata[tid] : 0; // 读共享内存
    if (warp == 0) {
        val = warp_reduce(val);
    }
    return val;
}

__global__ void reduce_v10(float *input, float *output, int n) {
    __shared__ float sdata[threadsPerBlock];

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
    // 这里不需要 __syncthreads()，因为 warp_reduce 中的 __*sync 函数
    // 保证 warp 内所有被 mask 选中的线同步才会继续执行
    sum = block_reduce(sum, sdata);
    if (tid == 0) {
        atomicAdd(output, sum);
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
    
    int numBlocks = (n + BLOCK_SIZE * COARSE_FACTOR - 1) / (BLOCK_SIZE * COARSE_FACTOR);

    float *input_d, *output_d;
    cudaMalloc(&input_d, n * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));

    cudaMemcpy(input_d, input_h, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_v10<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(input_d, output_d, n);

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