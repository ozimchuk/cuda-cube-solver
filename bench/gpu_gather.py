"""Random byte-gather throughput on the GPU, the access pattern of a pruning-table lookup."""
import sys
import time

import cupy as cp

KERNEL = r'''
extern "C" __global__ void gather64(const unsigned char* __restrict__ table,
                                    unsigned long long size, int iters,
                                    unsigned int* __restrict__ out) {
    unsigned long long x = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    x = x * 0x9E3779B97F4A7C15ull + 1;
    unsigned int acc = 0;
    for (int i = 0; i < iters; ++i) {
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        acc += table[__umul64hi(x, size)];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

extern "C" __global__ void gather32(const unsigned char* __restrict__ table,
                                    unsigned int words, int iters,
                                    unsigned int* __restrict__ out) {
    unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 1;
    unsigned int acc = 0;
    for (int i = 0; i < iters; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        unsigned long long idx = (unsigned long long)__umulhi(x, words) << 2;
        acc += table[idx + (x & 3)];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
'''


def run(kern, table, size_arg, blocks, threads, iters, out):
    args = (blocks,), (threads,), (table, size_arg, cp.int32(iters), out)
    kern(*args)
    cp.cuda.Device().synchronize()
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        kern(*args)
    cp.cuda.Device().synchronize()
    return reps * threads * blocks * iters / (time.perf_counter() - t)


def main():
    sizes_gib = [float(s) for s in sys.argv[1:]] or [0.25, 1, 4, 7.3, 14]
    module = cp.RawModule(code=KERNEL)
    k64, k32 = module.get_function("gather64"), module.get_function("gather32")
    threads, blocks, iters = 256, 36 * 64, 256
    out = cp.empty(threads * blocks, dtype=cp.uint32)
    for gib in sizes_gib:
        size = int(gib * 2**30) & ~3
        table = cp.empty(size, dtype=cp.uint8)
        table[::4096] = 1
        r64 = run(k64, table, cp.uint64(size), blocks, threads, iters, out)
        r32 = run(k32, table, cp.uint32(size // 4), blocks, threads, iters, out)
        print(f"GPU {gib:6.2f} GiB  hash64 {r64 / 1e9:6.3f}  xorshift32 {r32 / 1e9:6.3f} G lookups/s", flush=True)
        del table
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
