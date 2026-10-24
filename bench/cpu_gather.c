/* Random byte-gather throughput on the CPU with many independent loads in flight per thread. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define LANES 32

static uint8_t *table;
static uint64_t size;
static long iters;

static void *worker(void *arg) {
    uint64_t x[LANES];
    uint32_t acc = 0;
    for (int l = 0; l < LANES; ++l) x[l] = ((uint64_t)(uintptr_t)arg * LANES + l) * 0x9E3779B97F4A7C15ull + 1;
    for (long i = 0; i < iters; ++i)
        for (int l = 0; l < LANES; ++l) {
            uint64_t v = x[l];
            v ^= v >> 31; v *= 0xBF58476D1CE4E5B9ull; v ^= v >> 29;
            x[l] = v;
            acc += table[(unsigned __int128)v * size >> 64];
        }
    return (void *)(uintptr_t)acc;
}

int main(int argc, char **argv) {
    double gib = argc > 1 ? atof(argv[1]) : 4;
    int threads = argc > 2 ? atoi(argv[2]) : 12;
    iters = argc > 3 ? atol(argv[3]) : 2000000;
    size = (uint64_t)(gib * (1ull << 30));
    table = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(table, size, MADV_HUGEPAGE);
    memset(table, 1, size);
    pthread_t t[256];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < threads; ++i) pthread_create(&t[i], NULL, worker, (void *)(uintptr_t)i);
    for (int i = 0; i < threads; ++i) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double dt = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
    printf("CPU %6.2f GiB %2d threads  %7.3f G lookups/s\n", gib, threads, (double)threads * iters * LANES / dt / 1e9);
    return 0;
}
