#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define INLINE_FUNC __host__ __device__ __forceinline__

// 一般 libTorch 的类型都重载了运算符和std::numeric_limits，所以这里不需要对每个类型进行特化
// 但是如果你的T类型是 __half 或者 __nv_bfloat16，你需要特化这两个类型：
// template<>
// struct Sum<__half> {
//     INLINE_FUNC __half operator()(const __half &a, const __half &b) const {
//         return __hadd(a, b); // 注意如果 __CUDA_NO_HALF_OPERATORS__ 宏没定义不能直接用 a + b
//     }
//     INLINE_FUNC __half identity_element() const {
//         return CUDART_ZERO_FP16;
//     }
// };


template<typename T>
struct Sum {
    INLINE_FUNC T operator()(const T &a, const T &b) const {
        return a + b;
    }
    INLINE_FUNC T identity_element() const {
        return T(0);
    }
};

template<typename T>
struct Max {
    INLINE_FUNC T operator()(const T &a, const T &b) const {
        return a > b ? a : b;
    }
    INLINE_FUNC T identity_element() const {
        return T(std::numeric_limits<T>::min());
    }
};

template<typename T>
struct Min {
    INLINE_FUNC T operator()(const T &a, const T &b) const {
        return a < b ? a : b;
    }
    INLINE_FUNC T identity_element() const {
        return T(std::numeric_limits<T>::max());
    }
};