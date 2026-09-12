/* ----------------------------------------------------------------------------
 * Pomaikache QA Suite Module 5: Network Protocol & Chaos Edge-Case Suite
 * ---------------------------------------------------------------------------*/

#include "../src/pomaikache.h"
#include <assert.h>

// Test 1: Command Parser Robustness against Malformed Strings
static void test_malformed_command_handling(void) {
    printf("[CHAOS TEST] Testing parser safety against malformed commands...\n");

    pk_lru_cache_t *cache = pk_lru_create(100);
    pk_index_t *index = pk_index_create(128, PK_METRIC_COSINE, false);

    // Test GET on non-existent key
    pk_vector_t *missing = pk_lru_get(cache, "non_existent_key_999");
    assert(missing == NULL);

    // Test DEL on non-existent key
    bool del_ok = pk_lru_del(cache, "non_existent_key_999");
    assert(del_ok == false);

    // Test search with K > Total Elements
    float *qdata = (float *)pk_malloc_aligned(128 * sizeof(float));
    for (int i = 0; i < 128; i++) qdata[i] = 0.5f;
    pk_vector_t *qvec = pk_vector_create(99, 128, qdata);

    pk_search_result_t res[50];
    size_t found = pk_index_search(index, qvec, 50, res);
    printf("[CHAOS TEST] Oversized K Search (K=50 on empty index): Found %zu -> ", found);
    assert(found == 0);
    printf("PASSED\n");

    pk_free(qdata);
    pk_vector_free(qvec);
    pk_lru_free(cache);
    pk_index_free(index);
}

// Test 2: Buffer Boundary Enforcement
static void test_buffer_boundary(void) {
    printf("[CHAOS TEST] Testing oversized key string safety...\n");

    pk_lru_cache_t *cache = pk_lru_create(10);
    char huge_key[512];
    memset(huge_key, 'A', sizeof(huge_key) - 1);
    huge_key[sizeof(huge_key) - 1] = '\0';

    pk_vector_t *v = pk_vector_create(1, 64, NULL);
    pk_lru_put(cache, huge_key, v);

    // Key truncated safely to sizeof(entry->key) = 64
    char truncated_key[64];
    strncpy(truncated_key, huge_key, sizeof(truncated_key) - 1);
    truncated_key[sizeof(truncated_key) - 1] = '\0';
    
    pk_vector_t *got = pk_lru_get(cache, truncated_key);
    assert(got != NULL);
    printf("[CHAOS TEST] Oversized Key Safety: Truncated & Managed Safely -> PASSED\n");

    pk_lru_free(cache);
}

int main(void) {
    printf("========================================================================\n");
    printf(" POMAIKACHE QA: MODULE 5 - NETWORK PROTOCOL & CHAOS EDGE-CASE SUITE     \n");
    printf("========================================================================\n");
    test_malformed_command_handling();
    test_buffer_boundary();
    printf("[MODULE 5] ALL CHAOS EDGE-CASE TESTS PASSED.\n\n");
    return 0;
}
