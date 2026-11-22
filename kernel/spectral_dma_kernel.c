/*
 * =====================================================================================
 *  Dirichlet Regularization (dreg): Native C Zero-Copy Spectral DMA Micro-Kernel
 * =====================================================================================
 *  Platform: Cross-platform C11 (Win32 native events / POSIX pthreads)
 *  Features:
 *    - Static O(1) Base-3 Trit Lookup Table (256 x 5 int8 entries, 1.25 KB in L1 cache).
 *    - Direct zero-copy stream unpacking from .tritq bitstreams into caller-owned buffers.
 *    - SIMD-friendly i-k-j 2D-IDCT matrix multiplier (W = D_out^T * C * D_in).
 *    - Asynchronous double-buffered DMA prefetch engine on a dedicated OS worker thread.
 *    - Zero Python GIL contention; synchronized via Win32 Events or POSIX mutex/cond.
 * =====================================================================================
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32) && !defined(USE_POSIX_THREADS)
#include <windows.h>
#define MEMORY_BARRIER() MemoryBarrier()
#else
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#define MEMORY_BARRIER() __sync_synchronize()
#endif

#if defined(_MSC_VER)
#define DREG_RESTRICT __restrict
#else
#define DREG_RESTRICT restrict
#endif

#define MAX_SUBLAYERS 64
#define MAX_MATRICES_PER_SUBLAYER 8  /* 8 covers full Transformer block (4 attn + 3-4 FFN) */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* -------------------------------------------------------------------------
 * 1. Base-3 Trit Lookup Table (256 bytes -> 5 signed trits in {-1, 0, +1})
 * ------------------------------------------------------------------------- */
static int8_t TRIT_LUT[256][5];
static int g_trit_lut_initialized = 0;

static void init_trit_lut(void) {
    if (g_trit_lut_initialized) return;
    memset(TRIT_LUT, 0, sizeof(TRIT_LUT));
    for (int b = 0; b < 243; b++) {
        int val = b;
        for (int i = 0; i < 5; i++) {
            int rem = val % 3;
            TRIT_LUT[b][i] = (int8_t)(rem - 1);
            val /= 3;
        }
    }
    g_trit_lut_initialized = 1;
}

/* -------------------------------------------------------------------------
 * 2. SIMD-Friendly GEMM: C = A * B (i-k-j loop order for vectorization)
 * ------------------------------------------------------------------------- */
static inline void gemm_nn(
    const float* DREG_RESTRICT A,
    const float* DREG_RESTRICT B,
    float* DREG_RESTRICT C,
    int M, int K, int N
) {
    size_t c_bytes = (size_t)M * (size_t)N * sizeof(float);
    memset(C, 0, c_bytes);
    for (int i = 0; i < M; i++) {
        const float* a_row = &A[(size_t)i * (size_t)K];
        float* c_row = &C[(size_t)i * (size_t)N];
        for (int k = 0; k < K; k++) {
            float a_ik = a_row[k];
            const float* b_row = &B[(size_t)k * (size_t)N];
#if defined(__GNUC__) && !defined(__clang__)
            #pragma GCC ivdep
#endif
            for (int j = 0; j < N; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 3. Matrix & Sublayer Registry Definitions
 * ------------------------------------------------------------------------- */
typedef struct {
    int M;
    int N;
    int n0;
    int n1;
    int n2;
    float min0;
    float scale0;
    float min1;
    float scale1;
    float scale_trit;
    const uint8_t* b0;
    const uint8_t* b1;
    const uint8_t* b2;
    const int32_t* m0_idx;
    const int32_t* m1_idx;
    const int32_t* m2_idx;
    const float* D_out_T; /* Shape [M, M] */
    const float* D_in;    /* Shape [N, N] */
    size_t target_offset_floats;
} CMatrixRecord;

typedef struct {
    int num_matrices;
    CMatrixRecord matrices[MAX_MATRICES_PER_SUBLAYER];
} CSublayerRecord;

static CSublayerRecord g_sublayers[MAX_SUBLAYERS];
static int g_num_sublayers = 0;
static int g_initialized = 0;

/* -------------------------------------------------------------------------
 * 4. Core Matrix Decompression and 2D-IDCT Kernel
 * ------------------------------------------------------------------------- */
static inline void decode_single_matrix(
    const CMatrixRecord* rec,
    float* dct_buf,
    float* temp_buf,
    float* target_base_buf
) {
    int M = rec->M;
    int N = rec->N;
    if (M <= 0 || N <= 0) return;
    if (!rec->D_out_T || !rec->D_in || !dct_buf || !temp_buf || !target_base_buf) return;

    size_t total_el = (size_t)M * (size_t)N;

    /* Zero out the DCT workspace */
    memset(dct_buf, 0, total_el * sizeof(float));

    /* 1. Band 0: 8-bit linear uint8 */
    if (rec->n0 > 0 && rec->b0 != NULL && rec->m0_idx != NULL) {
        float min0 = rec->min0;
        float scale0 = rec->scale0;
        const uint8_t* b0 = rec->b0;
        const int32_t* m0 = rec->m0_idx;
        for (int i = 0; i < rec->n0; i++) {
            int32_t idx = m0[i];
            if (idx >= 0 && (size_t)idx < total_el) {
                dct_buf[idx] = ((float)b0[i]) * scale0 + min0;
            }
        }
    }

    /* 2. Band 1: 4-bit nibbles */
    if (rec->n1 > 0 && rec->b1 != NULL && rec->m1_idx != NULL) {
        float min1 = rec->min1;
        float scale1 = rec->scale1;
        const uint8_t* b1 = rec->b1;
        const int32_t* m1 = rec->m1_idx;
        int n1 = rec->n1;
        int n1_bytes = (n1 + 1) / 2;
        int out_i = 0;
        for (int i = 0; i < n1_bytes && out_i < n1; i++) {
            uint8_t byte = b1[i];
            uint8_t nib0 = (byte >> 4) & 0x0F;
            int32_t idx0 = m1[out_i++];
            if (idx0 >= 0 && (size_t)idx0 < total_el) {
                dct_buf[idx0] = ((float)nib0) * scale1 + min1;
            }
            if (out_i < n1) {
                uint8_t nib1 = byte & 0x0F;
                int32_t idx1 = m1[out_i++];
                if (idx1 >= 0 && (size_t)idx1 < total_el) {
                    dct_buf[idx1] = ((float)nib1) * scale1 + min1;
                }
            }
        }
    }

    /* 3. Band 2: Base-3 Ternary Trits via O(1) LUT */
    if (rec->n2 > 0 && rec->b2 != NULL && rec->m2_idx != NULL) {
        float scale_trit = rec->scale_trit;
        const uint8_t* b2 = rec->b2;
        const int32_t* m2 = rec->m2_idx;
        int n2 = rec->n2;
        int n2_bytes = (n2 + 4) / 5;
        int out_trit = 0;
        for (int i = 0; i < n2_bytes && out_trit < n2; i++) {
            uint8_t byte = b2[i];
            const int8_t* lut = TRIT_LUT[byte];
            for (int t = 0; t < 5 && out_trit < n2; t++) {
                int32_t idx = m2[out_trit++];
                if (idx >= 0 && (size_t)idx < total_el) {
                    dct_buf[idx] = ((float)lut[t]) * scale_trit;
                }
            }
        }
    }

    /* 4. 2D-IDCT:
     * temp_buf [M, N] = D_out_T [M, M] * dct_buf [M, N]
     * target_mat [M, N] = temp_buf [M, N] * D_in [N, N]
     */
    gemm_nn(rec->D_out_T, dct_buf, temp_buf, M, M, N);
    float* target_mat = target_base_buf + rec->target_offset_floats;
    gemm_nn(temp_buf, rec->D_in, target_mat, M, N, N);
}

static inline void decode_sublayer_internal(
    int k,
    float* target_sublayer_buf,
    float* dct_buf,
    float* temp_buf
) {
    if (k < 0 || k >= g_num_sublayers) return;
    const CSublayerRecord* sub = &g_sublayers[k];
    for (int m = 0; m < sub->num_matrices; m++) {
        if (sub->matrices[m].M > 0 && sub->matrices[m].N > 0) {
            decode_single_matrix(&sub->matrices[m], dct_buf, temp_buf, target_sublayer_buf);
        }
    }
}

/* -------------------------------------------------------------------------
 * 5. Asynchronous DMA Worker Thread (Win32 & POSIX Native, Zero-GIL)
 * ------------------------------------------------------------------------- */
static volatile int g_worker_stop = 0;

static int g_pending_k = -1;
static float* g_pending_target_buf = NULL;
static float* g_pending_dct_buf = NULL;
static float* g_pending_temp_buf = NULL;

#if defined(_WIN32) && !defined(USE_POSIX_THREADS)
static HANDLE g_h_req_event = NULL;
static HANDLE g_h_done_event = NULL;
static HANDLE g_h_worker_thread = NULL;

static DWORD WINAPI dma_worker_thread_func(LPVOID lpParam) {
    (void)lpParam;
    while (!g_worker_stop) {
        DWORD dwWait = WaitForSingleObject(g_h_req_event, INFINITE);
        if (dwWait != WAIT_OBJECT_0 || g_worker_stop) {
            break;
        }
        MEMORY_BARRIER();

        int k = g_pending_k;
        float* target_buf = g_pending_target_buf;
        float* dct_buf = g_pending_dct_buf;
        float* temp_buf = g_pending_temp_buf;

        if (k >= 0 && target_buf != NULL && dct_buf != NULL && temp_buf != NULL) {
            decode_sublayer_internal(k, target_buf, dct_buf, temp_buf);
        }

        MEMORY_BARRIER();
        SetEvent(g_h_done_event);
    }
    return 0;
}
#else
static pthread_t g_worker_thread;
static int g_worker_thread_started = 0;

static pthread_mutex_t g_req_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_req_cond  = PTHREAD_COND_INITIALIZER;
static int             g_req_flag  = 0;

static pthread_mutex_t g_done_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_done_cond  = PTHREAD_COND_INITIALIZER;
static int             g_done_flag  = 1;

static void* dma_worker_thread_posix(void* lpParam) {
    (void)lpParam;
    for (;;) {
        pthread_mutex_lock(&g_req_mutex);
        while (!g_req_flag && !g_worker_stop) {
            pthread_cond_wait(&g_req_cond, &g_req_mutex);
        }
        if (g_worker_stop) {
            pthread_mutex_unlock(&g_req_mutex);
            break;
        }
        g_req_flag = 0;
        int k = g_pending_k;
        float* target_buf = g_pending_target_buf;
        float* dct_buf = g_pending_dct_buf;
        float* temp_buf = g_pending_temp_buf;
        pthread_mutex_unlock(&g_req_mutex);

        if (k >= 0 && target_buf != NULL && dct_buf != NULL && temp_buf != NULL) {
            decode_sublayer_internal(k, target_buf, dct_buf, temp_buf);
        }

        pthread_mutex_lock(&g_done_mutex);
        g_done_flag = 1;
        pthread_cond_signal(&g_done_cond);
        pthread_mutex_unlock(&g_done_mutex);
    }
    return NULL;
}
#endif

/* -------------------------------------------------------------------------
 * 6. Exported C-FFI API
 * ------------------------------------------------------------------------- */

EXPORT void c_spectral_cleanup(void);

EXPORT int c_spectral_init(void) {
    if (g_initialized) {
        c_spectral_cleanup();
    }

    init_trit_lut();
    memset(g_sublayers, 0, sizeof(g_sublayers));
    g_num_sublayers = 0;
    g_worker_stop = 0;
    g_pending_k = -1;
    g_pending_target_buf = NULL;
    g_pending_dct_buf = NULL;
    g_pending_temp_buf = NULL;

#if defined(_WIN32) && !defined(USE_POSIX_THREADS)
    g_h_req_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    g_h_done_event = CreateEvent(NULL, FALSE, TRUE, NULL); /* Initially signaled */
    if (!g_h_req_event || !g_h_done_event) {
        if (g_h_req_event) { CloseHandle(g_h_req_event); g_h_req_event = NULL; }
        if (g_h_done_event) { CloseHandle(g_h_done_event); g_h_done_event = NULL; }
        return -1;
    }

    g_h_worker_thread = CreateThread(NULL, 0, dma_worker_thread_func, NULL, 0, NULL);
    if (!g_h_worker_thread) {
        CloseHandle(g_h_req_event); g_h_req_event = NULL;
        CloseHandle(g_h_done_event); g_h_done_event = NULL;
        return -2;
    }

    SetThreadPriority(g_h_worker_thread, THREAD_PRIORITY_ABOVE_NORMAL);
#else
    pthread_mutex_init(&g_req_mutex, NULL);
    pthread_cond_init(&g_req_cond, NULL);
    g_req_flag = 0;

    pthread_mutex_init(&g_done_mutex, NULL);
    pthread_cond_init(&g_done_cond, NULL);
    g_done_flag = 1;

    if (pthread_create(&g_worker_thread, NULL, dma_worker_thread_posix, NULL) != 0) {
        pthread_mutex_destroy(&g_req_mutex);
        pthread_cond_destroy(&g_req_cond);
        pthread_mutex_destroy(&g_done_mutex);
        pthread_cond_destroy(&g_done_cond);
        return -2;
    }
    g_worker_thread_started = 1;
#endif
    g_initialized = 1;
    return 0;
}

EXPORT void c_spectral_cleanup(void) {
    if (!g_initialized) return;

#if defined(_WIN32) && !defined(USE_POSIX_THREADS)
    g_worker_stop = 1;
    MEMORY_BARRIER();
    if (g_h_req_event) SetEvent(g_h_req_event);
    if (g_h_worker_thread) {
        WaitForSingleObject(g_h_worker_thread, 2000);
        CloseHandle(g_h_worker_thread);
        g_h_worker_thread = NULL;
    }
    if (g_h_req_event) { CloseHandle(g_h_req_event); g_h_req_event = NULL; }
    if (g_h_done_event) { CloseHandle(g_h_done_event); g_h_done_event = NULL; }
#else
    pthread_mutex_lock(&g_req_mutex);
    g_worker_stop = 1;
    g_req_flag = 1;
    pthread_cond_signal(&g_req_cond);
    pthread_mutex_unlock(&g_req_mutex);

    if (g_worker_thread_started) {
        pthread_join(g_worker_thread, NULL);
        g_worker_thread_started = 0;
    }
    pthread_mutex_destroy(&g_req_mutex);
    pthread_cond_destroy(&g_req_cond);
    pthread_mutex_destroy(&g_done_mutex);
    pthread_cond_destroy(&g_done_cond);
#endif
    memset(g_sublayers, 0, sizeof(g_sublayers));
    g_num_sublayers = 0;
    g_pending_k = -1;
    g_pending_target_buf = NULL;
    g_pending_dct_buf = NULL;
    g_pending_temp_buf = NULL;
    g_initialized = 0;
}

EXPORT int c_spectral_register_matrix(
    int sublayer_k,
    int mat_idx,
    int M, int N,
    int n0, int n1, int n2,
    float min0, float scale0,
    float min1, float scale1,
    float scale_trit,
    const uint8_t* b0,
    const uint8_t* b1,
    const uint8_t* b2,
    const int32_t* m0_idx,
    const int32_t* m1_idx,
    const int32_t* m2_idx,
    const float* D_out_T,
    const float* D_in,
    size_t target_offset_floats
) {
    if (sublayer_k < 0 || sublayer_k >= MAX_SUBLAYERS) return -1;
    if (mat_idx < 0 || mat_idx >= MAX_MATRICES_PER_SUBLAYER) {
        fprintf(stderr, "[c_spectral] ERROR: mat_idx %d >= MAX_MATRICES_PER_SUBLAYER %d (increase MAX or split sublayer)\n",
                mat_idx, MAX_MATRICES_PER_SUBLAYER);
        return -2;
    }
    if (M <= 0 || N <= 0) return -3;

    CSublayerRecord* sub = &g_sublayers[sublayer_k];
    if (mat_idx >= sub->num_matrices) {
        sub->num_matrices = mat_idx + 1;
    }
    if (sublayer_k >= g_num_sublayers) {
        g_num_sublayers = sublayer_k + 1;
    }

    CMatrixRecord* rec = &sub->matrices[mat_idx];
    rec->M = M;
    rec->N = N;
    rec->n0 = n0;
    rec->n1 = n1;
    rec->n2 = n2;
    rec->min0 = min0;
    rec->scale0 = scale0;
    rec->min1 = min1;
    rec->scale1 = scale1;
    rec->scale_trit = scale_trit;
    rec->b0 = b0;
    rec->b1 = b1;
    rec->b2 = b2;
    rec->m0_idx = m0_idx;
    rec->m1_idx = m1_idx;
    rec->m2_idx = m2_idx;
    rec->D_out_T = D_out_T;
    rec->D_in = D_in;
    rec->target_offset_floats = target_offset_floats;
    return 0;
}

EXPORT void c_spectral_decode_sublayer_sync(
    int k,
    float* target_sublayer_buf,
    float* dct_buf,
    float* temp_buf
) {
    init_trit_lut();
    decode_sublayer_internal(k, target_sublayer_buf, dct_buf, temp_buf);
}

EXPORT void c_spectral_dma_start_prefetch(
    int k,
    float* target_sublayer_buf,
    float* dct_buf,
    float* temp_buf
) {
    if (!g_initialized) return;

#if defined(_WIN32) && !defined(USE_POSIX_THREADS)
    ResetEvent(g_h_done_event);
    g_pending_k = k;
    g_pending_target_buf = target_sublayer_buf;
    g_pending_dct_buf = dct_buf;
    g_pending_temp_buf = temp_buf;
    MEMORY_BARRIER();
    SetEvent(g_h_req_event);
#else
    pthread_mutex_lock(&g_done_mutex);
    g_done_flag = 0;
    pthread_mutex_unlock(&g_done_mutex);

    pthread_mutex_lock(&g_req_mutex);
    g_pending_k = k;
    g_pending_target_buf = target_sublayer_buf;
    g_pending_dct_buf = dct_buf;
    g_pending_temp_buf = temp_buf;
    g_req_flag = 1;
    pthread_cond_signal(&g_req_cond);
    pthread_mutex_unlock(&g_req_mutex);
#endif
}

EXPORT void c_spectral_dma_wait_prefetch(void) {
    if (!g_initialized) return;

#if defined(_WIN32) && !defined(USE_POSIX_THREADS)
    WaitForSingleObject(g_h_done_event, INFINITE);
    MEMORY_BARRIER();
#else
    pthread_mutex_lock(&g_done_mutex);
    while (!g_done_flag) {
        pthread_cond_wait(&g_done_cond, &g_done_mutex);
    }
    pthread_mutex_unlock(&g_done_mutex);
#endif
}

/* Standalone decode for direct microbenchmarking of single matrix */
EXPORT void c_spectral_decode_matrix_direct(
    int M, int N,
    int n0, int n1, int n2,
    float min0, float scale0,
    float min1, float scale1,
    float scale_trit,
    const uint8_t* b0,
    const uint8_t* b1,
    const uint8_t* b2,
    const int32_t* m0_idx,
    const int32_t* m1_idx,
    const int32_t* m2_idx,
    const float* D_out_T,
    const float* D_in,
    float* dct_buf,
    float* temp_buf,
    float* target_buf
) {
    init_trit_lut();
    CMatrixRecord rec;
    rec.M = M;
    rec.N = N;
    rec.n0 = n0;
    rec.n1 = n1;
    rec.n2 = n2;
    rec.min0 = min0;
    rec.scale0 = scale0;
    rec.min1 = min1;
    rec.scale1 = scale1;
    rec.scale_trit = scale_trit;
    rec.b0 = b0;
    rec.b1 = b1;
    rec.b2 = b2;
    rec.m0_idx = m0_idx;
    rec.m1_idx = m1_idx;
    rec.m2_idx = m2_idx;
    rec.D_out_T = D_out_T;
    rec.D_in = D_in;
    rec.target_offset_floats = 0;

    decode_single_matrix(&rec, dct_buf, temp_buf, target_buf);
}
