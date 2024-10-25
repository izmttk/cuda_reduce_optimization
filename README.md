# Optimizing Reduce Kernel in CUDA Step by Step

## Learning Resources

### 官方文档

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Math API Reference Manual](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [reduction in CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction)

### 博客&教程

- [CUDA Mode](https://github.com/gpu-mode/lectures) 不错的视频课程，Lecture 9是关于reduction的，可以上手学习
- [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) 第2节讲reduction，另外还有他的[笔记](https://zhuanlan.zhihu.com/p/596012674)
- [PPT of "Optimizing Parallel Reduction in CUDA"](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 官方PDF优化教程，建议首先阅读
- [[CUDA 学习笔记] Reduce 算子优化](https://blog.csdn.net/LostUnravel/article/details/136104386)

与具体优化相关的资料会在小节中给出。

## Reduce Kernel

## reduce_v1

无优化，每个 thread 对应一个输入元素。
对应官方PDF Reduction #1中的Baseline版本。

## reduce_v2

缩减未使用的 thread，第一次迭代时数组范围内所有线程都进行read和reduce。每次迭代每个 thread 对应一次 reduce 操作。
对应官方PDF Reduction #4。

## reduce_v3

交错寻址（Interleaved Addressing），尽可能确保相邻的thread在相同的条件分支内，即尽可能确保一个 warp 内所有 thread 在相同分支内，缓解 warp divergence。
对应官方PDF Reduction #2。

- [PPT of "Lecture 3: control flow and synchronisation"](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec3.pdf)

## reduce_v4

调换相加的顺序，让一个 warp 内相邻的 thread 访存位置连续，这样能防止同一个 warp 内的 thread 在相同时刻访问相同 bank 的内存。减少 bank conflict。
对应官方PDF Reduction #3。

v4版本的顺序有问题，应该先使用 shared memory，再考虑减少 bank conflict。

- [How to Understand and Optimize Shared Memory Accesses using Nsight Compute](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/) 同时学学 Nsight Compute 怎么用

## reduce_v5

使用 shared memory，减少全局内存访问次数。但是每个 thread 读取一个全局内存数据。

## reduce_v6

使用 shared memory，同时在从全局内存读取数据时，进行一次 reduce 操作。减少空闲线程。
对应官方PDF Reduction #4。

## reduce_v7

线程粗化，thread coarsening，即一个 thread 读取并处理多个全局内存数据。
对应官方PDF Reduction #7。

## reduce_v8

循环展开最后一个 warp 的 reduce。
对应官方PDF Reduction #5。

## reduce_v9

使用 warp shuffle 指令，直接在寄存器内移动数据，减少 shared memory 的使用。

- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [PPT of "Warp Shuffle and Warp Vote Instructions"](https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2018.pdf)
- [How the Fermi Thread Block Scheduler Works](https://www.cs.rochester.edu/~sree/fermi-tbs/fermi-tbs.html) 帮助理解 warp scheduler 如何工作，如果你想看的话
- Xiao, Shucai, and Wu-chun Feng. "Inter-block GPU communication via fast barrier synchronization." 2010 IEEE International Symposium on Parallel & Distributed Processing (IPDPS). IEEE, 2010. [PDF](https://synergy.cs.vt.edu/pubs/papers/xiao-ipdps2010-gpusync.pdf) 两种 block 间通信的方法，但是注意无法避免死锁。
- 高岚,赵雨晨,张伟功,王晶,钱德沛.面向GPU并行编程的线程同步综述.软件学报,2024,35(2):1028-1047 [PDF](https://www.jos.org.cn/jos/article/pdf/6984) 理解为什么 CPU 的多线程同步方法在 GPU 中会死锁

## reduce_v10

Pytorch 的 block reduce 写法，即一个 block 内进行两次 warp reduce。

- [Pytorch 的 Block Reduce 实现](https://github.com/pytorch/pytorch/blob/245026af2d2f26c74993cb90e01bddbd627c6797/aten/src/ATen/native/cuda/block_reduce.cuh)
- [Oneflow 的 Block Reduce 实现](https://github.com/Oneflow-Inc/oneflow/blob/f0d13a6eb44d47a6288eba66e3cc777613bf9fde/oneflow/core/cuda/softmax.cuh#L56) 以及他们的[文章](https://zhuanlan.zhihu.com/p/341059988)

## reduce_v11

Pytorch 的 block reduce 写法，但是加点 C++ 模板，并且不限输入数据量。注意算法上不完全一样。

- [Pytorch Reduce 实现](https://github.com/pytorch/pytorch/blob/a77145ae2f48007eb6564584f1f4d2fdbb2570bd/aten/src/ATen/native/cuda/Reduce.cuh)
- [Achieved Occupancy](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm) 不知道哪里遗留下来的文档，帮助你理解 Occupancy 是怎么计算的
- [Requests, Wavefronts, Sectors Metrics: Understanding and Optimizing Memory-Bound Kernels with Nsight Compute](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/) 了解访存如何优化

## reduce_v12

Pytorch 的 block reduce 写法，使用向量化访存读取数据。

- [Pytorch 向量化访存的实现](https://github.com/pytorch/pytorch/blob/245026af2d2f26c74993cb90e01bddbd627c6797/aten/src/ATen/native/cuda/MemoryAccess.cuh#L158-L180)

## reduce_torch

使用 Pytorch 的 cpp extension 在 python 内调用自定义的 reduce kernel。

## TODO

- [ ] 使用 Makefile / CMake 管理编译
- [ ] 性能分析
