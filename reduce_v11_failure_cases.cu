// 1. 看似支持了所有blockSz情况，但实际上没什么用
// 因为一个block最多支持1024个线程，即32个warp，两次warp_reduce就解决了
template<typename T, typename Op = Sum, const int BlockSz = MAX_BLOCK_SIZE, const int WarpSz = WARP_SIZE>
__device__ __forceinline__ T block_reduce_2(T val, T* sdata, Op op = Op(), const T ident = T(0)) {
    const int tid = threadIdx.x;
    const int warp = tid / WarpSz;
    const int lane = tid % WarpSz;
    constexpr int maxNumWarps = div_ceiling(BlockSz, WarpSz);

#pragma unroll
    for (int nWarps = maxNumWarps; nWarps > 1; nWarps = div_ceiling(nWarps, WarpSz)) {
        if (warp < nWarps) {
            val = warp_reduce(val, op);
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
            val = tid < nWarps ? sdata[tid] : ident; // 读共享内存
        }
    }
    if (warp == 0) {
        val = warp_reduce(val, op);
    }
    return val;
}

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
template<typename T, typename Op, const int BlockSz, const int WarpSz>
__device__ __forceinline__ void reduce(T *input, T *output, int n, Op op, const T ident, T *block_output, int *counter) {
    // ......
    
    // 想进行block间同步，以便进行多轮block_reduce
    // 但实际上没什么用，这个操作的起因是block数量超过了1024个，因为一个block包含的线程是固定的
    // reduce里进行两次reduce，一次block内reduce，一次block间reduce，并且用的都是block_reduce
    // 于是最大支持的数据量是1024 * 1024 = 1048576，超过这个数据量范围的数据不会被计算，当然这是一个线程读一个数据的情况
    // 想要突破这个限制，就需要多轮block_reduce，但是这需要block间同步，于是出现了下面的几种错误尝试

    // 2. 看上去使用cooperative groups实现block同步，但实际上在大量数据下还会报错：
    // too many blocks in cooperative launch
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int maxNumBlocks = gridDim.x;
    int bid = blockIdx.x;
    cg::grid_group g = cg::this_grid();
    for (int nBlocks = maxNumBlocks; nBlocks > 1; nBlocks = div_ceiling(nBlocks, BlockSz)) {
        if (bid < nBlocks) {
            // 这里不需要 __syncthreads()，因为 warp_reduce 中的 __*sync 函数
            // 保证 warp 内所有被 mask 选中的线同步才会继续执行
            val = block_reduce_2<T, Op, BlockSz, WarpSz>(val, sdata, op, ident);
            if (tid == 0) {
                block_output[blockIdx.x] = val;
            }
            g.sync(); // block 同步
            val = (global_idx < nBlocks) ? block_output[global_idx] : ident;
        }
    }
    if (bid == 0) {
        val = block_reduce_2<T, Op, BlockSz, WarpSz>(val, sdata, op, ident);
        if (tid == 0) {
            output[0] = val;
        }
    }

    // 3. 自旋锁实现 block 同步，注意这里用锁实现 block 同步会导致死锁，因为由于资源限制，所有SM只能同时运行有限数量的block，
    // 例如我的4060Ti上最多同时并行816个block，并且只能等当前这个block运行完才能调度下一个block进SM
    // 如果释放锁的线程块还没有被调度进GPU，那么会造成正在被调度的线程块永远等待
    // 可以让一个线程块阻塞，但是不能让所有线程块阻塞，否则会出现上述死锁情况
    int bid = blockIdx.x;
    if (tid == 0) {
        if (bid == 0) {
            output[0] = ident;
        }
        // block sync
        atomicAdd(counter, 1);
        while (atomicCAS(counter, gridDim.x, gridDim.x) != gridDim.x) {}
        // mutex lock
        while (atomicCAS(lock, 0, 1) != 0) {}
        output[0] = op(val, output[0]);
        atomicCAS(lock, 1, 0);
    }
}

// 用了cooperative groups，就要用cudaLaunchCooperativeKernel启动kernel
template<typename T, typename Op>
void launch_reduce_kernel(T *input, T *output, int n, Op op, T ident) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int maxBlocks = deviceProp.maxBlocksPerMultiProcessor * deviceProp.multiProcessorCount;

    constexpr int blockSize = 1024;
    constexpr int warpSize = WARP_SIZE;
    int numBlocks = maxBlocks;

    T *block_output;
    int *counter;
    cudaMalloc(&block_output, numBlocks * sizeof(T));
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    // This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    printf("supportsCoopLaunch: %d\n", supportsCoopLaunch);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int maxBlocks = deviceProp.maxBlocksPerMultiProcessor * deviceProp.multiProcessorCount;
    std::cout << "Maximum blocks that can be run simultaneously: " << maxBlocks << std::endl;
    auto err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduce_kernel<T, Op, blockSize, warpSize>, blockSize, 0);
    if (err != cudaSuccess) {
        printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor error: %s\n", cudaGetErrorString(err));
    }
    printf("numBlocksPerSm: %d\n", numBlocksPerSm);
    // launch
    void *kernelArgs[] = { &input, &output, &n, &op, &ident, &block_output, &counter };
    dim3 dimBlock(blockSize, 1, 1);
    // dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    auto status = cudaLaunchCooperativeKernel((void*)reduce_kernel<T, Op, blockSize, warpSize>, dimGrid, dimBlock, kernelArgs);
    if (status != cudaSuccess) {
        printf("cudaLaunchCooperativeKernel error: %s\n", cudaGetErrorString(status));
    }
    cudaFree(block_output);
    cudaFree(counter);
}