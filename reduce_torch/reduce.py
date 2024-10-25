import torch
import time 
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
reduce_cpp = load(
    name='reduce',
    sources=[
        'reduce.cu',
    ],
    extra_cuda_cflags=[
        "-O2",
        '-std=c++20',
        '-Xcompiler', '/w',
        # "-U__CUDA_NO_HALF_OPERATORS__",
        # "-U__CUDA_NO_HALF_CONVERSIONS__",
        # "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    # verbose=True
)

def run_benchmark(perf_func: callable, values: torch.Tensor, tag: str, 
                  warmup: int = 10, iters: int = 100):
    # if perf_func.__name__ == torch.sum.__name__:
    #     values = values.float() # for precision
    for i in range(warmup):
        out = perf_func(values) # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    print(f"{out_info:>25}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


# Ss = [1024, 2048, 4096]
# Ks = [1024, 2048, 4096]
# SKs = [(S, K) for S in Ss for K in Ks]

# for (S, K) in SKs:
#     print("-" * 80)
#     print(f"S={S}, K={K}")
#     values = torch.randn((S, K)).cuda().float()
#     run_benchmark(reduce_cpp.sum, values, "lib_sum")
#     run_benchmark(torch.sum, values, "torch_sum")


values = torch.randn((1341003200,), device='cuda', dtype=torch.float16)
run_benchmark(reduce_cpp.sum, values, "custom_sum")
run_benchmark(torch.sum, values, "torch_sum")