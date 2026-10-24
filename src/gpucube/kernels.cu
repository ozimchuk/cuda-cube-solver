// GPU kernels for the 3x3 pruning table and IDA* search.
//
// Packed state: corners in two 32-bit words (byte = cp | co << 3),
// edges in three 32-bit words (byte = ep | eo << 4).
// Table: 2 bits per entry holding distance mod 3; 3 means unvisited.

typedef unsigned int u32;
typedef unsigned long long u64;

#define N_CCOMB 70u
#define N_CORNER_RAW 153090u

// Edge coordinate variant (vcube/nxopt naming): EP 1 = E-slice positions
// only, 2 = plus the order of the E edges (4!), 3 = plus which non-E
// positions hold U-layer edges (8C4); EO 4 = flips of the E edges, 8 = flips
// of the non-E edges, 12 = all flips.  The default variant is "208".
#ifndef EDGE_EP
#define EDGE_EP 2
#endif
#ifndef EDGE_EO
#define EDGE_EO 8
#endif
__host__ __device__ constexpr u32 n_ep(int ep) { return ep == 1 ? 1u : (ep == 2 ? 24u : 70u); }
__host__ __device__ constexpr u32 n_eo(int eo) { return eo == 4 ? 16u : (eo == 8 ? 256u : 2048u); }
template <int EP, int EO>
struct EdgeKind {
    static constexpr unsigned long long N = 495ull * n_ep(EP) * n_eo(EO);
};
#define N_EDGE (EdgeKind<EDGE_EP, EDGE_EO>::N)

struct Tables {
    u32 c_sel[18][2];     // prmt selectors for corner bytes
    u32 c_add[18][2];     // co << 3 added per byte
    u32 e_sel01[18][3];   // prmt selectors from e0:e1
    u32 e_sel2[18][3];    // prmt selectors from e2
    u32 e_mask[18][3];    // 0xff bytes take e2
    u32 e_flip[18][3];    // eo << 4 xored per byte
    u32 lc[18][2];        // left multiply: corner byte of move for each cubie
    u32 le[18][3];        // left multiply: edge byte of move for each cubie
    u32 cs_sel01[16][3];  // edge conjugation by sym: position permutation (as moves)
    u32 cs_sel2[16][3];
    u32 cs_mask[16][3];
    u32 cs_flip[16][3];   // flip of the source position, << 4 per byte
    u32 cs_rel[16][3];    // cubie relabel byte for each cubie: rel(c) | flip(c) << 4
    unsigned char s_src[16][12];   // conj: source position of new position i
    unsigned char s_rel[16][16];   // conj: cubie relabel
    unsigned char s_fsrc[16][12];  // conj: flip of source position
    unsigned char s_fcub[16][16];  // conj: flip of cubie home position
    unsigned char s_mul[16][16];   // index of M_t * M_s
    unsigned short ccomb_rank[256];
    unsigned char ccomb_unrank[70];
    unsigned char eorder_rank[256];
    unsigned char eorder_unrank[24];
    unsigned short slice_unrank[495];
    unsigned short slice_rank[4096];
    unsigned char pext6[64][64];   // pext6[mask][value]: bits of value selected by mask, compacted
    unsigned char compact4[16][256];  // 2-bit fields of a byte selected by a 4-bit mask, compacted
};

struct State {
    u32 c0, c1;
    u32 e0, e1, e2;
};

__device__ __forceinline__ void apply_move(const Tables& T, State& s, int m) {
    u32 a = __byte_perm(s.c0, s.c1, T.c_sel[m][0]) + T.c_add[m][0];
    u32 b = __byte_perm(s.c0, s.c1, T.c_sel[m][1]) + T.c_add[m][1];
    a -= __vcmpgeu4(a, 0x18181818u) & 0x18181818u;
    b -= __vcmpgeu4(b, 0x18181818u) & 0x18181818u;
    s.c0 = a;
    s.c1 = b;
    u32 out[3];
#pragma unroll
    for (int w = 0; w < 3; ++w) {
        u32 x = __byte_perm(s.e0, s.e1, T.e_sel01[m][w]);
        u32 y = __byte_perm(s.e2, s.e2, T.e_sel2[m][w]);
        out[w] = ((x & ~T.e_mask[m][w]) | (y & T.e_mask[m][w])) ^ T.e_flip[m][w];
    }
    s.e0 = out[0];
    s.e1 = out[1];
    s.e2 = out[2];
}

__device__ __forceinline__ u32 nibbles(u32 v) {
    v = (v | (v >> 4)) & 0x00FF00FFu;
    return (v | (v >> 8)) & 0xFFFFu;
}

// s <- compose(move m, s): relabel cubies through the move (used for inverses).
__device__ __forceinline__ void apply_left(const Tables& T, State& s, int m) {
    u32 a = __byte_perm(T.lc[m][0], T.lc[m][1], nibbles(s.c0 & 0x07070707u)) + (s.c0 & 0x18181818u);
    u32 b = __byte_perm(T.lc[m][0], T.lc[m][1], nibbles(s.c1 & 0x07070707u)) + (s.c1 & 0x18181818u);
    s.c0 = a - (__vcmpgeu4(a, 0x18181818u) & 0x18181818u);
    s.c1 = b - (__vcmpgeu4(b, 0x18181818u) & 0x18181818u);
    u32 e[3] = {s.e0, s.e1, s.e2};
#pragma unroll
    for (int w = 0; w < 3; ++w) {
        u32 v = e[w];
        u32 sel = nibbles(v & 0x07070707u);
        u32 x = __byte_perm(T.le[m][0], T.le[m][1], sel);
        u32 y = __byte_perm(T.le[m][2], T.le[m][2], sel);
        u32 hi = ((v >> 3) & 0x01010101u) * 0xFFu;
        e[w] = ((x & ~hi) | (y & hi)) ^ (v & 0x10101010u);
    }
    s.e0 = e[0];
    s.e1 = e[1];
    s.e2 = e[2];
}

__device__ __forceinline__ u32 gather_bit(u32 v, int bit) {
    v = (v >> bit) & 0x01010101u;
    return (v | (v >> 7) | (v >> 14) | (v >> 21)) & 0xFu;
}

__device__ __forceinline__ u32 corner_raw(const Tables& T, const State& s) {
    u32 olo = (s.c0 >> 3) & 0x03030303u;
    u32 ohi = (s.c1 >> 3) & 0x00030303u;
    u32 twist = __dp4a(olo, 0x1B090301u, 0u) + 81u * __dp4a(ohi, 0x00090301u, 0u);
    u32 dmask = gather_bit(s.c0, 2) | (gather_bit(s.c1, 2) << 4);
    return twist * N_CCOMB + T.ccomb_rank[dmask];
}

__device__ __forceinline__ u32 edge_byte(const State& s, int p) {
    u32 w = p < 4 ? s.e0 : (p < 8 ? s.e1 : s.e2);
    return (w >> ((p & 3) * 8)) & 0xFFu;
}

__device__ __forceinline__ u32 relabel_word(u32 v, u32 r0, u32 r1, u32 r2) {
    u32 sel = nibbles(v & 0x07070707u);
    u32 x = __byte_perm(r0, r1, sel);
    u32 y = __byte_perm(r2, r2, sel);
    u32 hi = ((v >> 3) & 0x01010101u) * 0xFFu;
    return ((x & ~hi) | (y & hi)) ^ (v & 0x10101010u);
}

__device__ __forceinline__ u32 labels8(u32 w) {
    u32 v = w & 0x03030303u;
    return (v | (v >> 6) | (v >> 12) | (v >> 18)) & 0xFFu;
}

__device__ __forceinline__ u32 pext12(const Tables& T, u32 value, u32 mask) {
    return T.pext6[mask & 63u][value & 63u] | ((u32)T.pext6[mask >> 6][(value >> 6) & 63u] << __popc(mask & 63u));
}

// Edge coordinate of M_sym g M_sym^-1, branch-free: permute positions and
// relabel cubies with byte permutes, then read the coordinate with bit tricks.
template <int EP, int EO>
__device__ __forceinline__ u32 edge_coord_k(const Tables& T, const State& s, int sym) {
    u32 e[3];
#pragma unroll
    for (int w = 0; w < 3; ++w) {
        u32 x = __byte_perm(s.e0, s.e1, T.cs_sel01[sym][w]);
        u32 y = __byte_perm(s.e2, s.e2, T.cs_sel2[sym][w]);
        u32 v = ((x & ~T.cs_mask[sym][w]) | (y & T.cs_mask[sym][w])) ^ T.cs_flip[sym][w];
        e[w] = relabel_word(v, T.cs_rel[sym][0], T.cs_rel[sym][1], T.cs_rel[sym][2]);
    }
    u32 m0 = gather_bit(e[0], 3), m1 = gather_bit(e[1], 3), m2 = gather_bit(e[2], 3);
    u32 emask = m0 | (m1 << 4) | (m2 << 8);
    u32 omask = gather_bit(e[0], 4) | (gather_bit(e[1], 4) << 4) | (gather_bit(e[2], 4) << 8);
    u32 nm = ~emask & 0xFFFu;
    u32 ep = 0;
    if (EP == 2) {
        u32 n0 = __popc(m0), n1 = __popc(m1);
        u32 eord = T.compact4[m0][labels8(e[0])] | ((u32)T.compact4[m1][labels8(e[1])] << (2 * n0)) |
                   ((u32)T.compact4[m2][labels8(e[2])] << (2 * (n0 + n1)));
        ep = T.eorder_rank[eord];
    } else if (EP == 3) {
        // U-layer edges are cubies 0..3: bits 2 and 3 of the byte clear.
        u32 umask = gather_bit(~(e[0] | (e[0] >> 1)), 2) | (gather_bit(~(e[1] | (e[1] >> 1)), 2) << 4) |
                    (gather_bit(~(e[2] | (e[2] >> 1)), 2) << 8);
        ep = T.ccomb_rank[pext12(T, umask, nm)];
    }
    u32 eo;
    if (EO == 4) eo = pext12(T, omask, emask);
    else if (EO == 8) eo = pext12(T, omask, nm);
    else eo = omask & 0x7FFu;
    return ((u32)T.slice_rank[emask] * n_ep(EP) + ep) * n_eo(EO) + eo;
}

__device__ __forceinline__ u32 edge_coord(const Tables& T, const State& s, int sym) {
    return edge_coord_k<EDGE_EP, EDGE_EO>(T, s, sym);
}

template <int EP, int EO>
__device__ __forceinline__ u64 table_index_k(const Tables& T, const u32* __restrict__ classidx, const State& s) {
    u32 packed = classidx[corner_raw(T, s)];
    return (u64)(packed >> 4) * EdgeKind<EP, EO>::N + edge_coord_k<EP, EO>(T, s, packed & 15u);
}

__device__ __forceinline__ u64 table_index(const Tables& T, const u32* __restrict__ classidx,
                                           const State& s) {
    return table_index_k<EDGE_EP, EDGE_EO>(T, classidx, s);
}

__device__ __forceinline__ u32 table_get(const u32* __restrict__ table, u64 idx) {
#ifdef PROBE_L2_ONLY
    idx &= 0x3FFFFFull;  // diagnostic: keep every lookup in a 1 MB region
#endif
    return (table[idx >> 4] >> ((idx & 15u) * 2u)) & 3u;
}

// Representative state of a table entry of the module's edge variant.
__device__ void unrank(const Tables& T, const u32* __restrict__ class_rep, u64 idx, State& s) {
    const u32 NEP = n_ep(EDGE_EP), NEO = n_eo(EDGE_EO);
    u32 cls = (u32)(idx / N_EDGE);
    u32 ecoord = (u32)(idx - (u64)cls * N_EDGE);
    u32 raw = class_rep[cls];
    u32 twist = raw / N_CCOMB;
    u32 dmask = T.ccomb_unrank[raw - twist * N_CCOMB];
    u32 c[8];
    u32 osum = 0, nu = 0, nd = 0;
    for (int i = 0; i < 8; ++i) {
        u32 o = 0;
        if (i < 7) {
            o = twist % 3u;
            twist /= 3u;
            osum += o;
        } else {
            o = (3u - osum % 3u) % 3u;
        }
        u32 cp = (dmask >> i) & 1u ? 4u + nd++ : nu++;
        c[i] = cp | (o << 3);
    }
    s.c0 = c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24);
    s.c1 = c[4] | (c[5] << 8) | (c[6] << 16) | (c[7] << 24);
    u32 eo = ecoord % NEO;
    u32 rest = ecoord / NEO;
    u32 ep = rest % NEP;
    u32 emask = T.slice_unrank[rest / NEP];
    u32 eord = EDGE_EP == 2 ? T.eorder_unrank[ep] : 0xE4u;  // identity order 0,1,2,3
    u32 umask = EDGE_EP == 3 ? T.ccomb_unrank[ep] : 0x0Fu;  // non-E slots holding U edges
    u32 e[3] = {0, 0, 0};
    u32 ne = 0, no = 0, nu_e = 0, nd_e = 0;
    for (int i = 0; i < 12; ++i) {
        u32 b, flip;
        if ((emask >> i) & 1u) {
            b = 8u | ((eord >> (2 * ne)) & 3u);
            flip = EDGE_EO == 4 ? (eo >> ne) & 1u : 0u;
            ++ne;
        } else {
            b = (umask >> no) & 1u ? nu_e++ : 4u + nd_e++;
            flip = EDGE_EO == 8 ? (eo >> no) & 1u : 0u;
            ++no;
        }
        if (EDGE_EO == 12) flip = i < 11 ? (eo >> i) & 1u : (u32)(__popc(eo) & 1);
        e[i >> 2] |= (b | (flip << 4)) << ((i & 3) * 8);
    }
    s.e0 = e[0];
    s.e1 = e[1];
    s.e2 = e[2];
}

__device__ __forceinline__ void load_tables(Tables& dst, const Tables* __restrict__ src) {
    const u32* s = reinterpret_cast<const u32*>(src);
    u32* d = reinterpret_cast<u32*>(&dst);
    for (int i = threadIdx.x; i < (int)(sizeof(Tables) / 4); i += blockDim.x) d[i] = s[i];
    __syncthreads();
}

// Set entry to value if it is unvisited; returns 1 if this call set it.
__device__ __forceinline__ u32 table_claim(u32* table, u64 idx, u32 value) {
    u32 shift = (idx & 15u) * 2u;
    u32 bit_clear = ((3u ^ value) << shift);
    u32 old = atomicAnd(&table[idx >> 4], ~bit_clear);
    return ((old >> shift) & 3u) == 3u;
}

// Block table: absolute lower bounds.  Each 32-byte sector holds an 8-bit
// block minimum in bits 0..7 of its first word, then 124 two-bit offsets
// from that minimum (3 means "at least minimum + 3").  One lookup touches
// one sector.
#define BLOCK_ENTRIES 124ull

__device__ __forceinline__ u32 block_lb(const u32* __restrict__ blk, u64 idx) {
#ifdef PROBE_L2_ONLY
    idx &= 0x3FFFFFull;
#endif
    u64 b = idx / BLOCK_ENTRIES;
    u32 pos = 8u + 2u * (u32)(idx - b * BLOCK_ENTRIES);
    const u32* p = blk + b * 8u;
    return (p[0] & 0xFFu) + ((p[pos >> 5] >> (pos & 31u)) & 3u);
}

// Record an entry first reached at depth.  BFS claims in increasing depth,
// so the first claim in a block fixes its minimum.
__device__ void block_set(u32* blk, u64 idx, u32 depth) {
    u64 b = idx / BLOCK_ENTRIES;
    u32 pos = 8u + 2u * (u32)(idx - b * BLOCK_ENTRIES);
    u32* p = blk + b * 8u;
    u32 w0 = *(volatile u32*)p;
    while ((w0 & 0xFFu) == 0xFFu) {
        u32 old = atomicCAS(p, w0, (w0 & ~0xFFu) | depth);
        if (old == w0) {
            w0 = (w0 & ~0xFFu) | depth;
            break;
        }
        w0 = old;
    }
    u32 v = min(depth - (w0 & 0xFFu), 3u);
    if (v) atomicOr(&p[pos >> 5], v << (pos & 31u));
}

// Mark a canonical neighbour and every stabilizer image of it at absolute
// depth in both the mod-3 table and the block table.
__device__ u32 claim_with_stabilizer(const Tables& T, u32* table, u32* blk,
                                     const u32* __restrict__ classidx,
                                     const unsigned short* __restrict__ class_stab,
                                     const State& n, u32 depth) {
    u32 value = depth % 3u;
    u32 packed = classidx[corner_raw(T, n)];
    u32 cls = packed >> 4;
    u32 sym = packed & 15u;
    u32 stab = class_stab[cls];
    u64 cls_base = (u64)cls * N_EDGE;
    u32 claimed = 0;
    while (stab) {
        u32 t = __ffs(stab) - 1;
        stab &= stab - 1;
        u64 idx = cls_base + edge_coord(T, n, T.s_mul[t][sym]);
        if (table_get(table, idx) == 3u && table_claim(table, idx, value)) {
            ++claimed;
            block_set(blk, idx, depth);
        }
    }
    return claimed;
}

extern "C" __global__ void tables_size(u32* out) { out[0] = sizeof(Tables); }

extern "C" __global__ void seed_solved(const Tables* __restrict__ gT, u32* table, u32* blk,
                                       const u32* __restrict__ classidx,
                                       const unsigned short* __restrict__ class_stab,
                                       u32* counter) {
    __shared__ Tables T;
    load_tables(T, gT);
    if (threadIdx.x) return;
    State s{0x03020100u, 0x07060504u, 0x03020100u, 0x07060504u, 0x0B0A0908u};
    atomicAdd(counter, claim_with_stabilizer(T, table, blk, classidx, class_stab, s, 0u));
}

// Forward step: expand every entry whose value is depth % 3.
extern "C" __global__ void expand_forward(const Tables* __restrict__ gT, u32* table, u32* blk,
                                          const u32* __restrict__ classidx,
                                          const u32* __restrict__ class_rep,
                                          const unsigned short* __restrict__ class_stab,
                                          u64 n_words, u64 n_entries, u32 depth,
                                          unsigned long long* counter) {
    __shared__ Tables T;
    load_tables(T, gT);
    u32 cur = depth % 3u;
    u32 pattern = cur * 0x55555555u;
    unsigned long long local = 0;
    for (u64 w = blockIdx.x * (u64)blockDim.x + threadIdx.x; w < n_words; w += (u64)gridDim.x * blockDim.x) {
        u32 word = table[w];
        u32 x = word ^ pattern;
        u32 hit = ~(x | (x >> 1)) & 0x55555555u;  // fields equal to cur
        while (hit) {
            u32 bit = __ffs(hit) - 1;
            hit &= hit - 1;
            u64 idx = w * 16u + bit / 2u;
            if (idx >= n_entries) break;
            State s;
            unrank(T, class_rep, idx, s);
            for (int m = 0; m < 18; ++m) {
                State n = s;
                apply_move(T, n, m);
                local += claim_with_stabilizer(T, table, blk, classidx, class_stab, n, depth + 1u);
            }
        }
    }
    atomicAdd(counter, local);
}

// Backward step: an unvisited entry with a neighbour at depth % 3 is at depth + 1.
extern "C" __global__ void expand_backward(const Tables* __restrict__ gT, u32* table, u32* blk,
                                           const u32* __restrict__ classidx,
                                           const u32* __restrict__ class_rep,
                                           u64 n_words, u64 n_entries, u32 depth,
                                           unsigned long long* counter) {
    __shared__ Tables T;
    load_tables(T, gT);
    u32 cur = depth % 3u, next = (depth + 1u) % 3u;
    unsigned long long local = 0;
    for (u64 w = blockIdx.x * (u64)blockDim.x + threadIdx.x; w < n_words; w += (u64)gridDim.x * blockDim.x) {
        u32 word = table[w];
        u32 hit = word & (word >> 1) & 0x55555555u;  // fields equal to 3
        u32 clear = 0;
        while (hit) {
            u32 bit = __ffs(hit) - 1;
            hit &= hit - 1;
            u64 idx = w * 16u + bit / 2u;
            if (idx >= n_entries) break;
            State s;
            unrank(T, class_rep, idx, s);
            for (int m = 0; m < 18; ++m) {
                State n = s;
                apply_move(T, n, m);
                if (table_get(table, table_index(T, classidx, n)) == cur) {
                    clear |= (3u ^ next) << bit;
                    block_set(blk, idx, depth + 1u);
                    ++local;
                    break;
                }
            }
        }
        if (clear) atomicAnd(&table[w], ~clear);
    }
    atomicAdd(counter, local);
}

// Histogram of 2-bit values.
extern "C" __global__ void count_values(const u32* __restrict__ table, u64 n_words, u64 n_entries,
                                        unsigned long long* counts) {
    unsigned long long local[4] = {0, 0, 0, 0};
    for (u64 w = blockIdx.x * (u64)blockDim.x + threadIdx.x; w < n_words; w += (u64)gridDim.x * blockDim.x) {
        u32 word = table[w];
        u32 valid = 16;
        if (w * 16u + 16u > n_entries) valid = (u32)(n_entries - w * 16u);
        for (u32 j = 0; j < valid; ++j) ++local[(word >> (2 * j)) & 3u];
    }
    for (int v = 0; v < 4; ++v) atomicAdd(&counts[v], local[v]);
}

extern "C" __global__ void probe_index(const Tables* __restrict__ gT, const u32* __restrict__ classidx,
                                       const u32* __restrict__ states, int n, int move,
                                       u64* __restrict__ out_idx, u32* __restrict__ out_state) {
    __shared__ Tables T;
    load_tables(T, gT);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    State s{states[5 * i], states[5 * i + 1], states[5 * i + 2], states[5 * i + 3], states[5 * i + 4]};
    if (move >= 0) apply_move(T, s, move);
    if (move <= -2) apply_left(T, s, -2 - move);
    out_idx[i] = table_index(T, classidx, s);
    out_state[5 * i] = s.c0;
    out_state[5 * i + 1] = s.c1;
    out_state[5 * i + 2] = s.e0;
    out_state[5 * i + 3] = s.e1;
    out_state[5 * i + 4] = s.e2;
}

// ---------------------------------------------------------------------------
// IDA* search.
//
// Each cube is searched in three frames: frame k holds R^k g R^-k, so the UD
// table measures the UD, RL and FB axes in turn.  Heuristic values are
// tracked exactly from the mod-3 table: a child's distance differs from its
// parent's by at most one.

struct SearchConst {
    unsigned char frame_move[3][18];  // move m in frame k
    unsigned char allowed[19][18];    // allowed[last + 1][m] for canonical sequences
};

__device__ __forceinline__ u32 lookup_exact(const Tables& T, const u32* __restrict__ table,
                                            const u32* __restrict__ classidx, const State& s, u32 parent) {
    u32 c = table_get(table, table_index(T, classidx, s));
    u32 r = (c + 3u - parent % 3u) % 3u;  // 0: same, 1: +1, 2: -1
    return r == 0u ? parent : (r == 1u ? parent + 1u : parent - 1u);
}

__device__ __forceinline__ int inverse_move(int m) { return 3 * (m / 3) + 2 - m % 3; }

__device__ __forceinline__ bool is_solved(const State& s) {
    return s.c0 == 0x03020100u && s.c1 == 0x07060504u && s.e0 == 0x03020100u &&
           s.e1 == 0x07060504u && s.e2 == 0x0B0A0908u;
}

// Exact table distance by descent: follow a neighbour whose value is one less mod 3.
extern "C" __global__ void root_distance(const Tables* __restrict__ gT, const u32* __restrict__ table,
                                         const u32* __restrict__ classidx,
                                         const u32* __restrict__ states, int n, u32* __restrict__ out) {
    __shared__ Tables T;
    load_tables(T, gT);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    State s{states[5 * i], states[5 * i + 1], states[5 * i + 2], states[5 * i + 3], states[5 * i + 4]};
    u32 d = 0;
    u32 v = table_get(table, table_index(T, classidx, s));
    for (;;) {
        u32 want = (v + 2u) % 3u;
        bool moved = false;
        for (int m = 0; m < 18 && !moved; ++m) {
            State t = s;
            apply_move(T, t, m);
            if (table_get(table, table_index(T, classidx, t)) == want) {
                s = t;
                v = want;
                moved = true;
            }
        }
        if (!moved) break;
        ++d;
    }
    out[i] = d;
}

#define MAX_DEPTH 24

struct CubeJob {
    State frame[6];         // 0..2: R^k g R^-k; 3..5: their inverses
    u32 h[6];
    u32 bound;
    u32 found;              // set to 1 by the first thread that finds a solution
    u32 solution_length;
    unsigned char solution[MAX_DEPTH];
    unsigned long long nodes;
};

// Lower bound from the block table.
// With the axis rule the value must bound the coset distance itself, so the
// true-distance bound parent - 1 is not mixed in.
__device__ __forceinline__ u32 lookup_block(const Tables& T, const u32* __restrict__ blk,
                                            const u32* __restrict__ classidx, const State& s, u32 parent) {
#ifdef NO_AXIS_RULE
    return max(block_lb(blk, table_index(T, classidx, s)), parent ? parent - 1u : 0u);
#else
    return block_lb(blk, table_index(T, classidx, s));
#endif
}

__device__ __forceinline__ u32 frame_lookup(const Tables& T, const u32* __restrict__ table,
                                            const u32* __restrict__ blk,
                                            const u32* __restrict__ classidx, const State& s, int k,
                                            u32 parent) {
#ifdef FORWARD_BLOCK
    // Forward frames from the block table: parent - 1 stays a coset bound
    // there because a right multiplication moves to an adjacent coset.
    if (k < 3) return max(block_lb(blk, table_index(T, classidx, s)), parent ? parent - 1u : 0u);
#endif
    return k < 3 ? lookup_exact(T, table, classidx, s, parent) : lookup_block(T, blk, classidx, s, parent);
}

// Block-table lower bounds for a list of packed states (validation).
extern "C" __global__ void probe_block(const Tables* __restrict__ gT, const u32* __restrict__ blk,
                                       const u32* __restrict__ classidx,
                                       const u32* __restrict__ states, int n, u32* __restrict__ out) {
    __shared__ Tables T;
    load_tables(T, gT);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    State s{states[5 * i], states[5 * i + 1], states[5 * i + 2], states[5 * i + 3], states[5 * i + 4]};
    out[i] = block_lb(blk, table_index(T, classidx, s));
}

// The table's subgroup contains every U and D turn, so a shortest coset
// representative never ends in U or D.  Hence if frame k (axis k after
// conjugation) has a lower bound equal to the remaining budget, no solution
// of exactly that length ends with a turn on axis k; for an inverse frame,
// none starts with one.  Faces U,R,F,D,L,B map to axis face % 3, which is
// the frame index.
//
// Axes (bit mask) on which the next move may not lie at a node.
template <int NF>
__device__ __forceinline__ u32 excluded_first_axes(const u32* h, u32 budget) {
#ifndef NO_AXIS_RULE
    if (NF == 6)
        return (u32)(h[3] == budget) | ((u32)(h[4] == budget) << 1) | ((u32)(h[5] == budget) << 2);
#endif
    return 0u;
}

// True if the child values rule out every last move (or every first move).
template <int NF>
__device__ __forceinline__ bool all_axes_tight(const u32* c, u32 budget) {
#ifndef NO_AXIS_RULE
    if (budget == 0) return false;  // the empty sequence has no last move
    if (c[0] == budget && c[1] == budget && c[2] == budget) return true;
    if (NF == 6 && c[3] == budget && c[4] == budget && c[5] == budget) return true;
#endif
    return false;
}

template <int NF>
__device__ __forceinline__ void frame_do(const Tables& T, const SearchConst& S, State& f, int k, int m) {
    if (k < 3) apply_move(T, f, S.frame_move[k][m]);
    else apply_left(T, f, inverse_move(S.frame_move[k - 3][m]));
}

template <int NF>
__device__ __forceinline__ void frame_undo(const Tables& T, const SearchConst& S, State& f, int k, int m) {
    if (k < 3) apply_move(T, f, inverse_move(S.frame_move[k][m]));
    else apply_left(T, f, S.frame_move[k - 3][m]);
}

// Work item w -> (cube w / n_prefix, prefix w % n_prefix).  Prefixes are
// canonical move sequences of length prefix_len stored as bytes.
template <int NF>
__device__ void search_impl(const Tables* __restrict__ gT, const SearchConst* __restrict__ gS,
                            const u32* __restrict__ table, const u32* __restrict__ blk,
                            const u32* __restrict__ classidx, CubeJob* jobs, const int* __restrict__ active, int n_active,
                            const unsigned char* __restrict__ prefixes, int n_prefix, int prefix_len,
                            unsigned long long* work_counter, unsigned long long* stats) {
    __shared__ Tables T;
    __shared__ SearchConst S;
    load_tables(T, gT);
    {
        const u32* src = reinterpret_cast<const u32*>(gS);
        u32* dst = reinterpret_cast<u32*>(&S);
        for (int i = threadIdx.x; i < (int)(sizeof(SearchConst) / 4); i += blockDim.x) dst[i] = src[i];
        __syncthreads();
    }
    unsigned char path[MAX_DEPTH];
    u32 hstack[MAX_DEPTH];
    const unsigned long long total = (unsigned long long)n_active * n_prefix;

    for (;;) {
        unsigned long long w = atomicAdd(work_counter, 1ull);
        if (w >= total) return;
        int job_id = active[w / n_prefix];
        CubeJob* job = &jobs[job_id];
        if (*(volatile u32*)&job->found) continue;
        const unsigned char* prefix = prefixes + (w % n_prefix) * prefix_len;
        const u32 bound = job->bound;
        State f[NF];
        u32 h[NF];
#pragma unroll
        for (int k = 0; k < NF; ++k) {
            f[k] = job->frame[k];
            h[k] = job->h[k];
        }
        unsigned long long nodes = 0;
        u32 st_look[NF], st_prune[NF], st_budget[4] = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < NF; ++k) st_look[k] = st_prune[k] = 0;
        int g = 0;
        int last = -1;
        bool ok = true;
        for (; g < prefix_len && ok; ++g) {
            int m = prefix[g];
            ++nodes;
            if ((excluded_first_axes<NF>(h, bound - g) >> ((m / 3) % 3)) & 1u) {
                ok = false;
                break;
            }
#pragma unroll
            for (int k = 0; k < NF; ++k) {
                frame_do<NF>(T, S, f[k], k, m);
                if (k != 3 + (m / 3) % 3) h[k] = frame_lookup(T, table, blk, classidx, f[k], k, h[k]);
                if (g + 1 + h[k] > bound) ok = false;
            }
            if (ok && all_axes_tight<NF>(h, bound - g - 1)) ok = false;
            path[g] = (unsigned char)m;
            last = m;
        }
        if (!ok) {
            atomicAdd(&job->nodes, nodes);
            continue;
        }
        const int base = g;
        int next = 0;
        u32 iter = 0;
        for (;;) {
            if ((++iter & 1023u) == 0 && *(volatile u32*)&job->found) break;
            if ((u32)g == bound) {
                if (is_solved(f[0]) && atomicCAS(&job->found, 0u, 1u) == 0u) {
                    job->solution_length = g;
                    for (int j = 0; j < g; ++j) job->solution[j] = path[j];
                }
                next = 18;
            }
            if (next < 18) {
                int m = next++;
                if (!S.allowed[last + 1][m]) continue;
                if ((excluded_first_axes<NF>(h, bound - g) >> ((m / 3) % 3)) & 1u) continue;
                ++nodes;
                ++st_budget[min(bound - g - 1u, 3u)];
                u32 c[NF];
                int pruned = NF;
                // A move on axis a leaves inverse frame a in the same coset
                // (the move becomes a U/D turn there, on the subgroup side).
                const int inherit = 3 + (m / 3) % 3;
#pragma unroll
                for (int k = 0; k < NF; ++k) {
                    if (pruned == NF) {
                        frame_do<NF>(T, S, f[k], k, m);
                        c[k] = k == inherit ? h[k] : frame_lookup(T, table, blk, classidx, f[k], k, h[k]);
                        ++st_look[k];
                        if (g + 1 + c[k] > bound) {
                            pruned = k;
                            ++st_prune[k];
                        }
                    }
                }
                if (pruned == NF && all_axes_tight<NF>(c, bound - g - 1)) pruned = NF - 1;
                if (pruned < NF) {
#pragma unroll
                    for (int k = 0; k < NF; ++k)
                        if (k <= pruned) frame_undo<NF>(T, S, f[k], k, m);
                    continue;
                }
                u32 packed = 0;
#pragma unroll
                for (int k = 0; k < NF; ++k) {
                    packed |= h[k] << (5 * k);
                    h[k] = c[k];
                }
                hstack[g] = packed;
                path[g] = (unsigned char)m;
                last = m;
                ++g;
                next = 0;
            } else {
                if (g == base) break;
                --g;
                int m = path[g];
#pragma unroll
                for (int k = 0; k < NF; ++k) {
                    frame_undo<NF>(T, S, f[k], k, m);
                    h[k] = (hstack[g] >> (5 * k)) & 31u;
                }
                next = m + 1;
                last = g > 0 ? path[g - 1] : -1;
            }
        }
        atomicAdd(&job->nodes, nodes);
        if (stats) {
#pragma unroll
            for (int k = 0; k < NF; ++k) {
                atomicAdd(&stats[k], (unsigned long long)st_look[k]);
                atomicAdd(&stats[8 + k], (unsigned long long)st_prune[k]);
            }
            for (int b = 0; b < 4; ++b) atomicAdd(&stats[16 + b], (unsigned long long)st_budget[b]);
        }
    }
}

#define SEARCH_ARGS const Tables* __restrict__ gT, const SearchConst* __restrict__ gS, \
    const u32* __restrict__ table, const u32* __restrict__ blk, \
    const u32* __restrict__ classidx, CubeJob* jobs, \
    const int* __restrict__ active, int n_active, const unsigned char* __restrict__ prefixes, \
    int n_prefix, int prefix_len, unsigned long long* work_counter, unsigned long long* stats
#define SEARCH_PASS gT, gS, table, blk, classidx, jobs, active, n_active, prefixes, n_prefix, prefix_len, work_counter, stats

#ifndef SEARCH_MIN_BLOCKS
#define SEARCH_MIN_BLOCKS 1
#endif
extern "C" __global__ void __launch_bounds__(256, SEARCH_MIN_BLOCKS) search3(SEARCH_ARGS) { search_impl<3>(SEARCH_PASS); }
extern "C" __global__ void __launch_bounds__(256, SEARCH_MIN_BLOCKS) search6(SEARCH_ARGS) { search_impl<6>(SEARCH_PASS); }

// ---------------------------------------------------------------------------
// Bidirectional IDA* (nxopt's search refinements, re-derived for this table).
//
// A node is the remaining problem p (find s with p s = I, |s| = budget) with
// six frames: R^k p R^-k (k = 0..2) and their inverses.  All values are
// coset lower bounds from the block table.
//
// * Extending the front with m: p <- p m.  Extending the back with m (the
//   solution's last remaining move): p <- m p.
// * A forward frame tight at the budget excludes last moves on its axis; an
//   inverse frame tight at the budget excludes first moves on its axis.  Each
//   node extends the end with more excluded axes (fewer moves to try).
// * A front move on axis a leaves inverse frame a in the same coset (and a
//   back move leaves forward frame a), so that value is inherited.
// * Moves on each end follow the canonical-sequence rule with the previous
//   move on that end; the solution is the front moves followed by the back
//   moves in reverse.

__device__ __forceinline__ int move_axis(int m) { return (m / 3) % 3; }

__device__ __forceinline__ void side_do(const Tables& T, const SearchConst& S, State& f, int k, int m, int side) {
    int mk = S.frame_move[k < 3 ? k : k - 3][m];
    if ((k < 3) == (side == 0)) apply_move(T, f, (k < 3) ? mk : inverse_move(mk));
    else apply_left(T, f, (k < 3) ? mk : inverse_move(mk));
}

__device__ __forceinline__ void side_undo(const Tables& T, const SearchConst& S, State& f, int k, int m, int side) {
    int mk = S.frame_move[k < 3 ? k : k - 3][m];
    if ((k < 3) == (side == 0)) apply_move(T, f, (k < 3) ? inverse_move(mk) : mk);
    else apply_left(T, f, (k < 3) ? inverse_move(mk) : mk);
}

__device__ __forceinline__ u32 tight_axes(const u32* h, int first, u32 budget) {
    return (u32)(h[first] == budget) | ((u32)(h[first + 1] == budget) << 1) | ((u32)(h[first + 2] == budget) << 2);
}

extern "C" __global__ void __launch_bounds__(256, SEARCH_MIN_BLOCKS)
search_bidir(const Tables* __restrict__ gT, const SearchConst* __restrict__ gS,
             const u32* __restrict__ blk, const u32* __restrict__ classidx,
             CubeJob* jobs, const int* __restrict__ active, int n_active,
             const unsigned char* __restrict__ prefixes, int n_prefix, int prefix_len,
             unsigned long long* work_counter, unsigned long long* stats) {
    __shared__ Tables T;
    __shared__ SearchConst S;
    load_tables(T, gT);
    {
        const u32* src = reinterpret_cast<const u32*>(gS);
        u32* dst = reinterpret_cast<u32*>(&S);
        for (int i = threadIdx.x; i < (int)(sizeof(SearchConst) / 4); i += blockDim.x) dst[i] = src[i];
        __syncthreads();
    }
    unsigned char path[MAX_DEPTH];   // move | side << 7
    unsigned char saved[MAX_DEPTH];  // previous end marker + 1 of the side that moved
    u32 hstack[MAX_DEPTH];
    const unsigned long long total = (unsigned long long)n_active * n_prefix;

    for (;;) {
        unsigned long long w = atomicAdd(work_counter, 1ull);
        if (w >= total) return;
        int job_id = active[w / n_prefix];
        CubeJob* job = &jobs[job_id];
        if (*(volatile u32*)&job->found) continue;
        const unsigned char* prefix = prefixes + (w % n_prefix) * prefix_len;
        const u32 bound = job->bound;
        State f[6];
        u32 h[6];
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            f[k] = job->frame[k];
            h[k] = job->h[k];
        }
        unsigned long long nodes = 0;
        int g = 0;
        int front_last = -1, back_first = -1;
        bool ok = true;
        for (; g < prefix_len && ok; ++g) {
            int m = prefix[g];
            ++nodes;
            int ax = move_axis(m);
            if ((tight_axes(h, 3, bound - g) >> ax) & 1u) {
                ok = false;
                break;
            }
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                side_do(T, S, f[k], k, m, 0);
                if (k != 3 + ax) h[k] = block_lb(blk, table_index(T, classidx, f[k]));
                if (g + 1 + h[k] > bound) ok = false;
            }
            u32 bc = bound - g - 1;
            if (ok && bc && (tight_axes(h, 0, bc) == 7u || tight_axes(h, 3, bc) == 7u)) ok = false;
            path[g] = (unsigned char)m;
            saved[g] = (unsigned char)(front_last + 1);
            front_last = m;
        }
        if (!ok) {
            atomicAdd(&job->nodes, nodes);
            continue;
        }
        const int base = g;
        int next = 0;
        u32 iter = 0;
        for (;;) {
            if ((++iter & 1023u) == 0 && *(volatile u32*)&job->found) break;
            const u32 budget = bound - g;
            if (budget == 0) {
                if (is_solved(f[0]) && atomicCAS(&job->found, 0u, 1u) == 0u) {
                    int n = 0;
                    for (int j = 0; j < g; ++j)
                        if (!(path[j] >> 7)) job->solution[n++] = path[j];
                    for (int j = g - 1; j >= 0; --j)
                        if (path[j] >> 7) job->solution[n++] = path[j] & 31u;
                    job->solution_length = g;
                }
                next = 18;
            }
            const u32 excl_first = tight_axes(h, 3, budget);
            const u32 excl_last = tight_axes(h, 0, budget);
            const int side = __popc(excl_last) > __popc(excl_first) ? 1 : 0;
            const u32 excl = side ? excl_last : excl_first;
            if (next < 18) {
                int m = next++;
                if (side == 0 ? !S.allowed[front_last + 1][m]
                              : (back_first >= 0 && !S.allowed[m + 1][back_first]))
                    continue;
                const int ax = move_axis(m);
                if ((excl >> ax) & 1u) continue;
                ++nodes;
                const int inherit = side == 0 ? 3 + ax : ax;
                u32 c[6];
                int pruned = 6;
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    if (pruned == 6) {
                        side_do(T, S, f[k], k, m, side);
                        c[k] = k == inherit ? h[k] : block_lb(blk, table_index(T, classidx, f[k]));
                        if (g + 1 + c[k] > bound) pruned = k;
                    }
                }
                const u32 bc = budget - 1;
                if (pruned == 6 && bc && (tight_axes(c, 0, bc) == 7u || tight_axes(c, 3, bc) == 7u)) pruned = 5;
                if (pruned < 6) {
#pragma unroll
                    for (int k = 0; k < 6; ++k)
                        if (k <= pruned) side_undo(T, S, f[k], k, m, side);
                    continue;
                }
                u32 packed = 0;
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    packed |= h[k] << (5 * k);
                    h[k] = c[k];
                }
                hstack[g] = packed;
                path[g] = (unsigned char)(m | (side << 7));
                if (side == 0) {
                    saved[g] = (unsigned char)(front_last + 1);
                    front_last = m;
                } else {
                    saved[g] = (unsigned char)(back_first + 1);
                    back_first = m;
                }
                ++g;
                next = 0;
            } else {
                if (g == base) break;
                --g;
                int m = path[g] & 31u;
                int s = path[g] >> 7;
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    side_undo(T, S, f[k], k, m, s);
                    h[k] = (hstack[g] >> (5 * k)) & 31u;
                }
                if (s == 0) front_last = (int)saved[g] - 1;
                else back_first = (int)saved[g] - 1;
                next = m + 1;
            }
        }
        atomicAdd(&job->nodes, nodes);
    }
}

// table_index(unrank(idx)) for a list of indices (validation; module's edge variant).
extern "C" __global__ void probe_unrank(const Tables* __restrict__ gT, const u32* __restrict__ classidx,
                                        const u32* __restrict__ class_rep, const u64* __restrict__ idx, int n,
                                        u64* __restrict__ out) {
    __shared__ Tables T;
    load_tables(T, gT);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    State s;
    unrank(T, class_rep, idx[i], s);
    out[i] = table_index(T, classidx, s);
}

// ---------------------------------------------------------------------------
// Multi-table search.  Every frame value is a coset lower bound from block
// tables: the main "208" table for all six frames first, then, only for
// children that survive it, the extra "112" and "304" tables.  Each table's
// subgroup contains every U and D turn, so the axis rule and the inverse-frame
// inheritance hold for the maximum over tables.

template <int K>
__device__ __forceinline__ u32 extra_lb(const Tables& T, const u32* __restrict__ classidx,
                                        const u32* __restrict__ x112, const u32* __restrict__ x304, const State& s) {
    return K == 0 ? block_lb(x112, table_index_k<1, 12>(T, classidx, s))
                  : block_lb(x304, table_index_k<3, 4>(T, classidx, s));
}

// Which frames consult the extra tables.  All by default: restricting to
// frames at the budget edge (EXTRA_EDGE_ONLY) loses most of the pruning,
// which comes through the axis rule and jumps of two or more.
#ifdef EXTRA_EDGE_ONLY
#define EXTRA_FRAME_OK(value, budget) ((value) == (budget))
#else
#define EXTRA_FRAME_OK(value, budget) true
#endif

template <int NX>
__device__ void search_multi_impl(const Tables* __restrict__ gT, const SearchConst* __restrict__ gS,
                                  const u32* __restrict__ blk, const u32* __restrict__ x112,
                                  const u32* __restrict__ x304, const u32* __restrict__ classidx,
                                  CubeJob* jobs, const int* __restrict__ active, int n_active,
                                  const unsigned char* __restrict__ prefixes, int n_prefix, int prefix_len,
                                  unsigned long long* work_counter) {
    __shared__ Tables T;
    __shared__ SearchConst S;
    load_tables(T, gT);
    {
        const u32* src = reinterpret_cast<const u32*>(gS);
        u32* dst = reinterpret_cast<u32*>(&S);
        for (int i = threadIdx.x; i < (int)(sizeof(SearchConst) / 4); i += blockDim.x) dst[i] = src[i];
        __syncthreads();
    }
    unsigned char path[MAX_DEPTH];
    u32 hstack[MAX_DEPTH];
    const unsigned long long total = (unsigned long long)n_active * n_prefix;

    // Value of frame k after the move: main table, forward frames also never
    // below parent - 1 (a right multiplication moves to an adjacent coset).
    auto main_lb = [&](const State& s, int k, u32 parent) -> u32 {
        u32 v = block_lb(blk, table_index(T, classidx, s));
        if (k < 3 && parent) v = max(v, parent - 1u);
        return v;
    };

    for (;;) {
        unsigned long long w = atomicAdd(work_counter, 1ull);
        if (w >= total) return;
        int job_id = active[w / n_prefix];
        CubeJob* job = &jobs[job_id];
        if (*(volatile u32*)&job->found) continue;
        const unsigned char* prefix = prefixes + (w % n_prefix) * prefix_len;
        const u32 bound = job->bound;
        State f[6];
        u32 h[6];
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            f[k] = job->frame[k];
            h[k] = job->h[k];
        }
        unsigned long long nodes = 0;
        int g = 0;
        int last = -1;
        bool ok = true;
        for (; g < prefix_len && ok; ++g) {
            int m = prefix[g];
            ++nodes;
            if ((excluded_first_axes<6>(h, bound - g) >> ((m / 3) % 3)) & 1u) {
                ok = false;
                break;
            }
            const int inherit = 3 + (m / 3) % 3;
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                frame_do<6>(T, S, f[k], k, m);
                if (k != inherit) {
                    u32 v = main_lb(f[k], k, h[k]);
                    if (NX > 0) v = max(v, extra_lb<0>(T, classidx, x112, x304, f[k]));
                    if (NX > 1) v = max(v, extra_lb<1>(T, classidx, x112, x304, f[k]));
                    h[k] = v;
                }
                if (g + 1 + h[k] > bound) ok = false;
            }
            if (ok && all_axes_tight<6>(h, bound - g - 1)) ok = false;
            path[g] = (unsigned char)m;
            last = m;
        }
        if (!ok) {
            atomicAdd(&job->nodes, nodes);
            continue;
        }
        const int base = g;
        int next = 0;
        u32 iter = 0;
        for (;;) {
            if ((++iter & 1023u) == 0 && *(volatile u32*)&job->found) break;
            if ((u32)g == bound) {
                if (is_solved(f[0]) && atomicCAS(&job->found, 0u, 1u) == 0u) {
                    job->solution_length = g;
                    for (int j = 0; j < g; ++j) job->solution[j] = path[j];
                }
                next = 18;
            }
            if (next < 18) {
                int m = next++;
                if (!S.allowed[last + 1][m]) continue;
                if ((excluded_first_axes<6>(h, bound - g) >> ((m / 3) % 3)) & 1u) continue;
                ++nodes;
                const int inherit = 3 + (m / 3) % 3;
                u32 c[6];
                int pruned = 6;
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    if (pruned == 6) {
                        frame_do<6>(T, S, f[k], k, m);
                        c[k] = k == inherit ? h[k] : main_lb(f[k], k, h[k]);
                        if (g + 1 + c[k] > bound) pruned = k;
                    }
                }
                if (NX > 0) {
#pragma unroll
                    for (int k = 0; k < 6; ++k) {
                        if (pruned == 6 && k != inherit && EXTRA_FRAME_OK(c[k], bound - g - 1)) {
                            c[k] = max(c[k], extra_lb<0>(T, classidx, x112, x304, f[k]));
                            if (g + 1 + c[k] > bound) pruned = 5;
                        }
                    }
                }
                if (NX > 1) {
#pragma unroll
                    for (int k = 0; k < 6; ++k) {
                        if (pruned == 6 && k != inherit && EXTRA_FRAME_OK(c[k], bound - g - 1)) {
                            c[k] = max(c[k], extra_lb<1>(T, classidx, x112, x304, f[k]));
                            if (g + 1 + c[k] > bound) pruned = 5;
                        }
                    }
                }
                if (pruned == 6 && all_axes_tight<6>(c, bound - g - 1)) pruned = 5;
                if (pruned < 6) {
#pragma unroll
                    for (int k = 0; k < 6; ++k)
                        if (k <= pruned) frame_undo<6>(T, S, f[k], k, m);
                    continue;
                }
                u32 packed = 0;
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    packed |= h[k] << (5 * k);
                    h[k] = c[k];
                }
                hstack[g] = packed;
                path[g] = (unsigned char)m;
                last = m;
                ++g;
                next = 0;
            } else {
                if (g == base) break;
                --g;
                int m = path[g];
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    frame_undo<6>(T, S, f[k], k, m);
                    h[k] = (hstack[g] >> (5 * k)) & 31u;
                }
                next = m + 1;
                last = g > 0 ? path[g - 1] : -1;
            }
        }
        atomicAdd(&job->nodes, nodes);
    }
}

#define MULTI_ARGS const Tables* __restrict__ gT, const SearchConst* __restrict__ gS, \
    const u32* __restrict__ blk, const u32* __restrict__ x112, const u32* __restrict__ x304, \
    const u32* __restrict__ classidx, CubeJob* jobs, const int* __restrict__ active, int n_active, \
    const unsigned char* __restrict__ prefixes, int n_prefix, int prefix_len, unsigned long long* work_counter
#define MULTI_PASS gT, gS, blk, x112, x304, classidx, jobs, active, n_active, prefixes, n_prefix, prefix_len, work_counter

extern "C" __global__ void __launch_bounds__(256, SEARCH_MIN_BLOCKS) search_multi0(MULTI_ARGS) { search_multi_impl<0>(MULTI_PASS); }
extern "C" __global__ void __launch_bounds__(256, SEARCH_MIN_BLOCKS) search_multi1(MULTI_ARGS) { search_multi_impl<1>(MULTI_PASS); }
extern "C" __global__ void __launch_bounds__(256, SEARCH_MIN_BLOCKS) search_multi2(MULTI_ARGS) { search_multi_impl<2>(MULTI_PASS); }
