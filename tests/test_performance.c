/* ----------------------------------------------------------------------------
 * Pomaikache QA Suite Module 6: Performance Regression & Tail Latency Suite
 * ---------------------------------------------------------------------------*/

#include "../src/pomaikache.h"
#include <assert.h>
#include <time.h>

#define PERF_VECTORS 5000
#define PERF_QUERIES 1000
#define PERF_DIM 128

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static void test_tail_latency_distribution(void) {
    printf("[PERF TEST] Running tail latency distribution profiling over %d queries...\n", PERF_QUERIES);

    pk_index_t *index = pk_index_create(PERF_DIM, PK_METRIC_COSINE, false);
    float *buf = (float *)pk_malloc_aligned(PERF_DIM * sizeof(float));

    for (size_t i = 0; i < PERF_VECTORS; i++) {
        for (uint32_t d = 0; d < PERF_DIM; d++) buf[d] = (float)rand() / (float)RAND_MAX;
        pk_vector_t *vec = pk_vector_create((uint64_t)(i + 1), PERF_DIM, buf);
        pk_index_insert(index, vec);
    }

    pk_vector_t *query = pk_vector_create(9999, PERF_DIM, buf);
    pk_search_result_t res[10];

    float *latencies = (float *)pk_malloc(PERF_QUERIES * sizeof(float));

    for (size_t q = 0; q < PERF_QUERIES; q++) {
        double start = get_time_sec();
        pk_index_search(index, query, 10, res);
        double elapsed = get_time_sec() - start;
        latencies[q] = (float)(elapsed * 1e6); // us
    }

    qsort(latencies, PERF_QUERIES, sizeof(float), cmp_float);

    float p50  = latencies[(size_t)(PERF_QUERIES * 0.50)];
    float p95  = latencies[(size_t)(PERF_QUERIES * 0.95)];
    float p99  = latencies[(size_t)(PERF_QUERIES * 0.99)];
    float p999 = latencies[(size_t)(PERF_QUERIES * 0.999)];

    printf("[PERF TEST] p50  (Median) : %8.2f us\n", p50);
    printf("[PERF TEST] p95           : %8.2f us\n", p95);
    printf("[PERF TEST] p99  (Tail)   : %8.2f us\n", p99);
    printf("[PERF TEST] p99.9 (Strict): %8.2f us\n", p999);

    assert(p50 < 5000.0f); // Assert median latency remains under 5ms

    pk_vector_free(query);
    pk_free(buf);
    pk_free(latencies);
    pk_index_free(index);
}

int main(void) {
    printf("========================================================================\n");
    printf(" POMAIKACHE QA: MODULE 6 - PERFORMANCE & TAIL LATENCY BENCHMARK        \n");
    printf("========================================================================\n");
    test_tail_latency_distribution();
    printf("[MODULE 6] ALL PERFORMANCE TESTS PASSED.\n\n");
    return 0;
}
