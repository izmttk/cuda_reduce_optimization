#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// pytorch 的 block reduce 写法
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024

struct Sum {
    template<typename T>
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
    }
};

struct Max {
    template<typename T>
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a > b ? a : b;
    }
};

struct Min {
    template<typename T>
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? a : b;
    }
};

template<typename T>
__host__ __device__ __forceinline__ constexpr T div_ceiling(T a, T b) {
    return (a + b - 1) / b;
}

// 一个 warp 有 32 个线程
// 全部 32 个线程都会被 reduce，不判断边界
template<typename T, typename Op = Sum, const int WarpSz = WARP_SIZE>
__device__ __forceinline__ T warp_reduce(T val, Op op = Op()) {
#pragma unroll
    for (int offset = WarpSz / 2 ; offset > 0; offset /= 2) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 一个 block 最多 1024 个线程，即 32 个 warp
template<typename T, typename Op = Sum, const int BlockSz = MAX_BLOCK_SIZE, const int WarpSz = WARP_SIZE>
__device__ __forceinline__ T block_reduce(T val, T* sdata, Op op = Op(), const T ident = T(0)) {
    const int tid = threadIdx.x;
    const int warp = tid / WarpSz;
    const int lane = tid % WarpSz;
    constexpr int maxNumWarps = div_ceiling(BlockSz, WarpSz);

    val = warp_reduce(val, op);
    // __syncthreads();
    // 如果要连续调用 block_reduce，需要在此处或者两个函数调用之间同步
    // 否则如果 sdata 相同，会发生读写冲突，举例如下：
    // sum1 = BlockReduceSum(val1, shared);
    // sum2 = BlockReduceSum(val2, shared);
    // 1、假设有block中有2个warp，在第二次__syncthreads()之后，warp1执行较快结束进入第二个BlockReduceSum中
    // 2、此时 warp0 正准备读取shared[1]，但warp1先在第二个BlockReduceSum中写入shared[1]，这就出现了读写冲突
    if (lane == 0) { // 这里会发生 warp divergence，无法避免
        // 为什么在 ncu 中这里会观察到少量 bank conflict？
        sdata[warp] = val; // 写共享内存
    }
    // 同样在共享内存读写之间需要同步
    // 否则第1个 warp 在写完共享内存后开始读取了，其他 warp 还没写完
    __syncthreads();
    val = (tid < maxNumWarps) ? sdata[tid] : ident; // 读共享内存
    if (warp == 0) {
        val = warp_reduce<T, Op, maxNumWarps>(val, op);
    }
    return val;
}

// Reference:
// https://github.com/pytorch/pytorch/blob/042f2f7746a064f1527d95d1f1d712b4f0b34186/aten/src/ATen/native/cuda/Reduce.cuh#L689
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions

template<typename T, typename Op, const int BlockSz, const int WarpSz>
__device__ __forceinline__ void reduce(T *input, T *output, int n, Op op, const T ident, T *block_output, int *counter) {
    constexpr int maxNumWarps = div_ceiling(BlockSz, WarpSz);
    __shared__ T sdata[maxNumWarps];

    int tid = threadIdx.x;

    T val = ident;

    int numThreads = gridDim.x * blockDim.x; // 网格中所有线程数

    // 反正最大并行度摆在这了，直接把输入规模缩减到 gridDim.x * blockDim.x，理论计算时间应该是不变的
    int numTiles = div_ceiling(n, numThreads);
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 循环展开在这里没什么用
    for (int tile = 0; tile < numTiles; tile++) {
        int idx = input_idx + tile * numThreads;
        if (idx < n) {
            val = op(val, input[idx]);
        }
    }
    
    // 这里不需要 __syncthreads()，因为 warp_reduce 中的 __*sync 函数
    // 保证 warp 内所有被 mask 选中的线同步才会继续执行
    val = block_reduce<T, Op, BlockSz, WarpSz>(val, sdata, op, ident);

    __shared__ bool is_last_block_done; // 注意该变量全 block 共享
    if (tid == 0) {
        block_output[blockIdx.x] = val;

        // 接下来是 block 同步机制
        
        // 首先确保 count 的写入发生在 output 写入之后、
        // 这样只要执行了 atomicAdd，就表示 output 写入操作已经完成
        __threadfence();

        // 每有一个 block 完成，count + 1
        int prev_block_finished = atomicAdd(counter, 1);
        // 所有 block 完成后 prev_block_finished == gridDim.x - 1
        is_last_block_done = (prev_block_finished == gridDim.x - 1);
    }
    // is_last_block_done 由每个块的第一个线程写入
    // 防止其他线程在第一个线程写入之前读取
    __syncthreads();
    // 最后一步由最后一个写入 output 的 block 完成
    if (is_last_block_done) {
        val = (tid < gridDim.x) ? block_output[tid] : ident;
        val = block_reduce<T, Op, BlockSz, WarpSz>(val, sdata, op, ident);
        if (tid == 0) {
            output[0] = val;
        }
    }
}

// 仅仅是一个包装函数
template<typename T, typename Op, const int BlockSz, const int WarpSz>
__global__ void reduce_kernel(T *input, T *output, int n, Op op, const T ident, T *block_output, int *counter) {
    reduce<T, Op, BlockSz, WarpSz>(input, output, n, op, ident, block_output, counter);
}

template<typename T, typename Op>
void launch_reduce_kernel(T *input, T *output, int n, Op op, T ident) {
    constexpr int blockSize = 512;
    constexpr int warpSize = WARP_SIZE;

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    // GPU占用率 = active warps / supported maximum active warps
    int maxActiveBlocks =  deviceProp.maxThreadsPerMultiProcessor / blockSize * deviceProp.multiProcessorCount;
    printf("maxThreadsPerMultiProcessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
    printf("maxActiveBlocks: %d\n", maxActiveBlocks);
    // 让block数不超过最大活动block数
    int numBlocks = min(maxActiveBlocks, div_ceiling(n, blockSize));
    // int numBlocks = div_ceiling(n, blockSize);
    T *block_output;
    int *counter;
    cudaMalloc(&block_output, numBlocks * sizeof(T));
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));
    printf("Launch config: grid(%d, 1, 1), block(%d, 1, 1)\n", numBlocks, blockSize);
    reduce_kernel<T, Op, blockSize, warpSize><<<numBlocks, blockSize>>>(input, output, n, op, ident, block_output, counter);
    cudaFree(block_output);
    cudaFree(counter);
}

template<typename T>
void reduce_sum(T *input, T *output, int n) {
    Sum op = Sum();
    T ident = T(0);
    launch_reduce_kernel(input, output, n, op, ident);
}

template<typename T>
void reduce_max(T *input, T *output, int n) {
    Max op = Max();
    T ident = std::numeric_limits<T>::min();
    launch_reduce_kernel(input, output, n, op, ident);
}

template<typename T>
void reduce_min(T *input, T *output, int n) {
    Min op = Min();
    T ident = std::numeric_limits<T>::max();
    launch_reduce_kernel(input, output, n, op, ident);
}

template<typename T>
T *generate_data(int n) {
    srand(time(NULL)); // time.h 包含在了 cuda_runtime.h 中
    T *data = new T[n];
    for (int i = 0; i < n; i++) {
        data[i] = rand() % 100;
    }
    return data;
}

using Type = float; // 测试类型，请注意浮点类型会因为精度问题导致测试失败

int main() {
    int n = 0;
    std::cout << "Input n: ";
    std::cin >> n;
    // 一些重要的输入规模
    // 只有一个 block： 1 << 4 + 1 = 17
    // 多于一个 block 但是小于最大并行 block 数：1 << 11 + 1 = 2049
    // 多余最大并行 block 数：1 << 22 + 1 = 4194305
    Type *input_h = generate_data<Type>(n);
    Type *output_h = new Type;

    Type *input_d, *output_d;
    cudaMalloc(&input_d, n * sizeof(Type));
    cudaMalloc(&output_d, sizeof(Type));

    cudaMemcpy(input_d, input_h, n * sizeof(Type), cudaMemcpyHostToDevice);

    // test sum
    reduce_sum(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, sizeof(Type), cudaMemcpyDeviceToHost);
    Type sum = std::reduce(input_h, input_h + n);
    std::cout << "sum (cpu): " << sum << std::endl;
    std::cout << "sum (gpu): " << *output_h << std::endl;

    if (sum == *output_h) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // test max
    reduce_max(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, sizeof(Type), cudaMemcpyDeviceToHost);
    Type max = std::reduce(input_h, input_h + n, std::numeric_limits<Type>::min(), [](Type a, Type b) { return a > b ? a : b; });
    std::cout << "max (cpu): " << max << std::endl;
    std::cout << "max (gpu): " << *output_h << std::endl;

    if (max == *output_h) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // test min
    reduce_min(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, sizeof(Type), cudaMemcpyDeviceToHost);
    Type min = std::reduce(input_h, input_h + n, std::numeric_limits<Type>::max(), [](Type a, Type b) { return a < b ? a : b; });    // c++17 特性
    std::cout << "min (cpu): " << min << std::endl;
    std::cout << "min (gpu): " << *output_h << std::endl;

    if (min == *output_h) {
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