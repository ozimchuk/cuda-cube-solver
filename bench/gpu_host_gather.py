"""Random byte reads from GPU kernels into pinned (mapped) host memory over PCIe."""
import sys
import time

import cupy as cp
import numpy as np

KERNEL = r'''
extern "C" __global__ void gather(const unsigned char* __restrict__ table, unsigned int words, int iters,
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


def main():
    gib = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    size = int(gib * 2**30) & ~3
    t = time.perf_counter()
    host = cp.cuda.alloc_pinned_memory(size)
    arr = np.frombuffer(host, dtype=np.uint8, count=size)
    arr[::4096] = 1
    print(f"pinned {gib} GiB in {time.perf_counter() - t:.1f}s", flush=True)
    kern = cp.RawKernel(KERNEL, "gather")
    ptr = host.ptr  # unified addressing: the host pointer is valid on the device
    for blocks, threads, iters in [(36 * 8, 256, 64), (36 * 32, 256, 64), (36 * 64, 256, 32)]:
        out = cp.empty(blocks * threads, dtype=cp.uint32)
        args = (blocks,), (threads,), (ptr, cp.uint32(size // 4), cp.int32(iters), out)
        kern(*args)
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        kern(*args)
        cp.cuda.Device().synchronize()
        dt = time.perf_counter() - t
        print(f"{blocks * threads:7d} threads: {blocks * threads * iters / dt / 1e6:8.1f} M random host reads/s", flush=True)


if __name__ == "__main__":
    main()
