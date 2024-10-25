#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/ctime>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "reduce_ops.cuh"
#include <type_traits>



// 向量化访存
// pytorch 的 block reduce 写法
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024


template<typename T>
__host__ __device__ __forceinline__ constexpr T div_ceiling(T a, T b) {
    return (a + b - 1) / b;
}

// 一个 warp 有 32 个线程
// 全部 32 个线程都会被 reduce，不判断边界
template<typename T, typename Op, const int WarpSz = WARP_SIZE>
__device__ __forceinline__ T warp_reduce(T val, Op op = Op()) {
    // 某些类型（比如c10::Half）可能同时支持转换到 float 和 __half，会造成 __shfl_down_sync 重载匹配的歧义；
    // 于是对于那些支持转换到 __half 和 __nv_bfloat16 的类型，我们假设转换到这两种类型是最高效精确的，比如c10::Half；
    // 我们就直接使用  __half 和 __nv_bfloat16 类型的 __shfl_down_sync 重载，而不是 float 版本；
    // 需要确保别的类型不会隐式转换到这两种类型（定义了 operator __half() 之类的就会导致隐式类型转换）；
    // 可以使用 __CUDA_NO_HALF_CONVERSIONS__ 宏关闭内置类型到 __half 和 __nv_bfloat16 的隐式转换；
    // 保险起见，先判断是否是基础类型，如果是，不管能不能隐式转换，都直接使用基础类型。
    using shlf_t = std::conditional_t<std::is_fundamental_v<T>, T,
        std::conditional_t<std::is_convertible_v<T, __half>, __half, 
        std::conditional_t<std::is_convertible_v<T, __nv_bfloat16>, __nv_bfloat16, T>>>;
#pragma unroll
    for (int offset = WarpSz / 2 ; offset > 0; offset /= 2) {
        // 低版本 ncu 会在这里看到 100 多个 live registers，是 bug，已在 CUDA 12.4 中修复
        val = op(val,  static_cast<T>(__shfl_down_sync(0xffffffff, static_cast<shlf_t>(val), offset)));
    }
    return val;
}

// https://github.com/pytorch/pytorch/blob/120fbe9caa49da42a84c3ef01c108b96adf4d9ac/aten/src/ATen/native/cuda/MemoryAccess.cuh#L158-L180
// aligned vector generates vectorized load/store on CUDA
template<typename T, int VecSz>
struct alignas(sizeof(T) * VecSz) aligned_vector {
    T val[VecSz];
    __device__ __forceinline__ aligned_vector() {};
    __device__ __forceinline__ aligned_vector(T v) {
#pragma unroll
        for (int i = 0; i < VecSz; i++) {
            val[i] = v;
        }
    }
    __device__ __forceinline__ T &operator[](uint32_t i) {
        return val[i];
    }
};
template <typename T, int VecSz>
__device__ __forceinline__ aligned_vector<T, VecSz> vec_load(const T *base_ptr, uint32_t offset) {
    using vec_t = aligned_vector<T, VecSz>;
    auto *from = reinterpret_cast<const vec_t *>(base_ptr);
    return from[offset];
}
template<typename T, int VecSz, typename Op>
__device__ __forceinline__ void vec_op(aligned_vector<T, VecSz> &a, aligned_vector<T, VecSz> &b, Op op) {
#pragma unroll
    for (int i = 0; i < VecSz; i++) {
        a[i] = op(a[i], b[i]);
    }
}
template<typename T, int VecSz, typename Op>
__device__ __forceinline__ T vec_reduce(T val, aligned_vector<T, VecSz> &a, Op op) {
#pragma unroll
    for (int i = 0; i < VecSz; i++) {
        val = op(val, a[i]);
    }
    return val;
}

// 一个 block 最多 1024 个线程，即 32 个 warp
template<typename T, typename Op, const int BlockSz = MAX_BLOCK_SIZE, const int WarpSz = WARP_SIZE>
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

template<typename T, typename Op, const int VecSz, const int BlockSz, const int WarpSz>
__device__ __forceinline__ void reduce(T *input, T *output, int n, Op op, const T ident, T *block_output, int *counter) {
    constexpr int maxNumWarps = div_ceiling(BlockSz, WarpSz);
    __shared__ T sdata[maxNumWarps];

    int tid = threadIdx.x;

    T val = ident;

    // 反正最大并行度摆在这了，直接把输入规模缩减到 gridDim.x * blockDim.x，理论计算时间应该是不变的
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // 网格中所有线程数
    // 向量化访存
    aligned_vector<T, VecSz> reduce_vec(ident);
    // 使用reduce_v11中的for循环模式会导致产生4个LDG.E.128指令，为什么？
    for (int i = input_idx; i * VecSz + VecSz <= n; i += stride) {
        auto vec = vec_load<T, VecSz>(input, i);
        vec_op(reduce_vec, vec, op);
    }
    val = vec_reduce(val, reduce_vec, op);
    // 剩余无法构成一个完整的向量的元素
    int tail_start = n - n % VecSz;
    int tail_idx = tail_start + threadIdx.x;
    if (tail_idx < n) {
        val = op(val, input[tail_idx]);
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
template<typename T, typename Op, const int VecSz, const int BlockSz, const int WarpSz>
__global__ void reduce_kernel(T *input, T *output, int n, Op op, const T ident, T *block_output, int *counter) {
    reduce<T, Op, VecSz, BlockSz, WarpSz>(input, output, n, op, ident, block_output, counter);
}