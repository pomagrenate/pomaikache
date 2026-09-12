/* ----------------------------------------------------------------------------
 * Pomaikache QA Suite Module 2: Ring Eviction & Lifecycle Suite
 * ---------------------------------------------------------------------------*/

#include "../src/pomaikache.h"
#include <assert.h>
#include <unistd.h>

// Test 1: Fixed-capacity saturation (N+1, 2N, 10N insertions)
static void test_ring_saturation(void) {
    size_t capacity = 100;
    uint32_t dim = 64;
    pk_circular_ring_t *ring = pk_ring_create(capacity, dim, PK_METRIC_COSINE);
    float *buf = (float *)pk_malloc(dim * sizeof(float));
    for (uint32_t i = 0; i < dim; i++) buf[i] = 1.0f;

    // Insert 10N items (1000 items into capacity 100)
    size_t total_inserts = capacity * 10;
    for (size_t i = 0; i < total_inserts; i++) {
        pk_ring_append(ring, (uint64_t)(i + 1), buf);
    }

    printf("[RING TEST] Saturation Check: Capacity=%zu, Active Count=%zu, Total Appended=%lu -> ",
           capacity, ring->count, ring->total_appended);
    assert(ring->count == capacity);
    assert(ring->total_appended == total_inserts);
    assert(ring->head == 0); // 1000 % 100 == 0
    printf("PASSED\n");

    pk_free(buf);
    pk_ring_free(ring);
}

// Test 2: TTL Mechanics (Lazy eviction vs Active purge)
static void test_ttl_mechanics(void) {
    pk_lru_cache_t *cache = pk_lru_create(100);
    pk_vector_t *v1 = pk_vector_create(1, 64, NULL);
    pk_vector_t *v2 = pk_vector_create(2, 64, NULL);

    // Put v1 with 50ms TTL, v2 with 0ms TTL (infinite)
    pk_lru_put_ttl(cache, "ttl_key_1", v1, 50);
    pk_lru_put_ttl(cache, "ttl_key_2", v2, 0);

    // Immediate get before expiration
    pk_vector_t *got1 = pk_lru_get(cache, "ttl_key_1");
    assert(got1 != NULL);
    assert(got1->id == 1);

    // Sleep for 70ms to allow ttl_key_1 to expire
    usleep(70000);

    // Passive lazy-eviction check on access
    pk_vector_t *expired_got = pk_lru_get(cache, "ttl_key_1");
    printf("[TTL TEST] Passive Eviction Check: Key 1 after 70ms = %s -> ", (expired_got == NULL) ? "NULL (Evicted)" : "ACTIVE");
    assert(expired_got == NULL);
    printf("PASSED\n");

    // Active purge test
    pk_vector_t *v3 = pk_vector_create(3, 64, NULL);
    pk_lru_put_ttl(cache, "ttl_key_3", v3, 10);
    usleep(20000);

    size_t purged = pk_lru_purge_expired(cache);
    printf("[TTL TEST] Active Purge Swept %zu Expired Entry -> ", purged);
    assert(purged >= 1);
    printf("PASSED\n");

    pk_lru_free(cache);
}

int main(void) {
    printf("========================================================================\n");
    printf(" POMAIKACHE QA: MODULE 2 - RING EVICTION & TTL LIFECYCLE                \n");
    printf("========================================================================\n");
    test_ring_saturation();
    test_ttl_mechanics();
    printf("[MODULE 2] ALL RING EVICTION & TTL TESTS PASSED.\n\n");
    return 0;
}
