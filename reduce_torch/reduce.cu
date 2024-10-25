#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include "reduce.cuh"

template<typename T, typename Op>
void launch_reduce_kernel(T *input, T *output, int n, Op op, T ident) {

    constexpr int vecSize = 128 / (8 * sizeof(T)); // LDG.128
    constexpr int blockSize = 512;
    constexpr int warpSize = C10_WARP_SIZE;

    const int maxThreadsPerMultiProcessor = at::cuda::getCurrentDeviceProperties() -> maxThreadsPerMultiProcessor;
    const int multiProcessorCount = at::cuda::getCurrentDeviceProperties() -> multiProcessorCount;
    // GPU占用率 = active warps / supported maximum active warps
    const int maxActiveBlocks =  maxThreadsPerMultiProcessor / blockSize * multiProcessorCount;

    // 让block数不超过最大活动block数
    const int numBlocks = min(maxActiveBlocks, div_ceiling(n, blockSize * vecSize));

    const int block_output_size = numBlocks * sizeof(T);
    const int semaphores_size = sizeof(int);
    
    auto& allocator = *c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr block_output = allocator.allocate(block_output_size);
    c10::DataPtr semaphores = allocator.allocate(semaphores_size);
    cudaMemsetAsync(semaphores.get(), 0, semaphores_size);
    reduce_kernel<T, Op, vecSize, blockSize, warpSize><<<numBlocks, blockSize>>>(input, output, n, op, ident, (T*)block_output.get(), (int*)semaphores.get());
}


template<typename T, template<typename> class Op, typename DeviceT = T>
void tensor_reduce(torch::Tensor input, torch::Tensor output) {
    Op op = Op<DeviceT>();
    DeviceT ident = op.identity_element();
    auto input_ptr = reinterpret_cast<DeviceT*>(input.data_ptr<T>());
    auto output_ptr = reinterpret_cast<DeviceT*>(output.data_ptr<T>());
    launch_reduce_kernel(input_ptr, output_ptr, input.numel(), op, ident);
}

#define DISPATCH_FLOAT_INTIGER_BOOL(...)          \
    AT_DISPATCH_CASE(c10::kFloat, __VA_ARGS__)    \
    AT_DISPATCH_CASE(c10::kDouble, __VA_ARGS__)   \
    AT_DISPATCH_CASE(c10::kChar, __VA_ARGS__)     \
    AT_DISPATCH_CASE(c10::kInt, __VA_ARGS__)      \
    AT_DISPATCH_CASE(c10::kLong, __VA_ARGS__)     \
    AT_DISPATCH_CASE(c10::kShort, __VA_ARGS__)    \
    AT_DISPATCH_CASE(c10::kByte, __VA_ARGS__)     \
    AT_DISPATCH_CASE(c10::kUInt16, __VA_ARGS__)   \
    AT_DISPATCH_CASE(c10::kUInt32, __VA_ARGS__)   \
    AT_DISPATCH_CASE(c10::kUInt64, __VA_ARGS__)   \
    AT_DISPATCH_CASE(c10::kBool, __VA_ARGS__)     \
    AT_DISPATCH_CASE(c10::kHalf, __VA_ARGS__)     \
    AT_DISPATCH_CASE(c10::kBFloat16, __VA_ARGS__)

torch::Tensor sum_forward(torch::Tensor x) {
    torch::Tensor out = torch::empty({1}, x.options());
    AT_DISPATCH_SWITCH(x.scalar_type(), "sum_forward", 
        DISPATCH_FLOAT_INTIGER_BOOL([&] {
            tensor_reduce<scalar_t, Sum>(x, out);
        })
    );
    return out;
}

torch::Tensor max_forward(torch::Tensor x) {
    torch::Tensor out = torch::empty({1}, x.options());
    AT_DISPATCH_SWITCH(x.scalar_type(), "max_forward", 
        DISPATCH_FLOAT_INTIGER_BOOL([&] {
            tensor_reduce<scalar_t, Max>(x, out);
        })
    );
    return out;
}

torch::Tensor min_forward(torch::Tensor x) {
    torch::Tensor out = torch::empty({1}, x.options());
    AT_DISPATCH_SWITCH(x.scalar_type(), "min_forward", 
        DISPATCH_FLOAT_INTIGER_BOOL([&] {
            tensor_reduce<scalar_t, Min>(x, out);
        })
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum", torch::wrap_pybind_function(sum_forward), "custom reduce sum");
    m.def("max", torch::wrap_pybind_function(max_forward), "custom reduce max");
    m.def("min", torch::wrap_pybind_function(min_forward), "custom reduce min");
}

// TODO
// 删了cuda_bf16.h试试
// 清理头文件
// 