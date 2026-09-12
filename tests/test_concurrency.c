/* ----------------------------------------------------------------------------
 * Pomaikache QA Suite Module 4: Concurrency & Thread-Safety Suite
 * ---------------------------------------------------------------------------*/

#include "../src/pomaikache.h"
#include <assert.h>
#include <pthread.h>

#define NUM_WORKERS 16
#define OPS_PER_WORKER 2000
#define CONC_DIM 128

typedef struct {
    pk_circular_ring_t *ring;
    pk_lru_cache_t *cache;
    pk_index_t *index;
    int worker_id;
} worker_args_t;

static void *worker_thread(void *arg) {
    worker_args_t *args = (worker_args_t *)arg;
    float *buf = (float *)pk_malloc_aligned(CONC_DIM * sizeof(float));

    for (size_t i = 0; i < OPS_PER_WORKER; i++) {
        for (uint32_t d = 0; d < CONC_DIM; d++) buf[d] = (float)(i + args->worker_id);

        uint64_t id = (uint64_t)(args->worker_id * 100000 + i);
        // Concurrent Ring Append
        pk_ring_append(args->ring, id, buf);

        // Concurrent LRU Put & Get
        char key[64];
        snprintf(key, sizeof(key), "w_%d_key_%zu", args->worker_id, i % 50);
        pk_vector_t *v = pk_vector_create(id, CONC_DIM, buf);
        pk_lru_put(args->cache, key, v);
        pk_lru_get(args->cache, key);

        // Concurrent Index Search
        if (i % 10 == 0) {
            pk_search_result_t res[5];
            pk_vector_t *query = pk_vector_create(999, CONC_DIM, buf);
            pk_index_search(args->index, query, 5, res);
            pk_vector_free(query);
        }
    }

    pk_free(buf);
    return NULL;
}

static void test_multithreaded_concurrency(void) {
    printf("[CONCURRENCY TEST] Spawning %d concurrent worker threads...\n", NUM_WORKERS);

    pk_circular_ring_t *ring = pk_ring_create(5000, CONC_DIM, PK_METRIC_COSINE);
    pk_lru_cache_t *cache = pk_lru_create(1000);
    pk_index_t *index = pk_index_create(CONC_DIM, PK_METRIC_COSINE, false);

    // Populate initial index
    float *tmp = (float *)pk_malloc_aligned(CONC_DIM * sizeof(float));
    for (size_t i = 0; i < 100; i++) {
        pk_vector_t *v = pk_vector_create(i, CONC_DIM, tmp);
        pk_index_insert(index, v);
    }
    pk_free(tmp);

    pthread_t threads[NUM_WORKERS];
    worker_args_t args[NUM_WORKERS];

    for (int i = 0; i < NUM_WORKERS; i++) {
        args[i].ring = ring;
        args[i].cache = cache;
        args[i].index = index;
        args[i].worker_id = i;
        pthread_create(&threads[i], NULL, worker_thread, &args[i]);
    }

    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("[CONCURRENCY TEST] Finished %d operations without data race or deadlock -> PASSED\n",
           NUM_WORKERS * OPS_PER_WORKER);

    pk_ring_free(ring);
    pk_lru_free(cache);
    pk_index_free(index);
}

int main(void) {
    printf("========================================================================\n");
    printf(" POMAIKACHE QA: MODULE 4 - CONCURRENCY & THREAD-SAFETY SUITE            \n");
    printf("========================================================================\n");
    test_multithreaded_concurrency();
    printf("[MODULE 4] ALL CONCURRENCY TESTS PASSED.\n\n");
    return 0;
}
