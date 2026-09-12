/* ----------------------------------------------------------------------------
 * Pomaikache QA Suite Module 3: Memory Safety & 64-Byte Alignment Suite
 * ---------------------------------------------------------------------------*/

#include "../src/pomaikache.h"
#include <assert.h>

#define NUM_ALLOCS 5000
#define TEST_DIM 128

// Test 1: Programmatic 64-byte alignment assertion for AVX2/AVX-512 SIMD
static void test_64byte_alignment(void) {
    printf("[MEM TEST] Asserting 64-byte SIMD alignment across %d allocations...\n", NUM_ALLOCS);

    float *ptrs[NUM_ALLOCS];
    for (int i = 0; i < NUM_ALLOCS; i++) {
        ptrs[i] = (float *)pk_malloc_aligned(TEST_DIM * sizeof(float));
        uintptr_t addr = (uintptr_t)ptrs[i];
        assert((addr % 64) == 0); // Must strictly satisfy 64-byte alignment
    }

    printf("[MEM TEST] 64-Byte Alignment Assertion: 100%% PASSED\n");

    for (int i = 0; i < NUM_ALLOCS; i++) {
        pk_free(ptrs[i]);
    }
}

// Test 2: Vector Struct Allocation & Vector Data Alignment
static void test_vector_data_alignment(void) {
    pk_vector_t *vecs[100];
    for (int i = 0; i < 100; i++) {
        vecs[i] = pk_vector_create((uint64_t)i, TEST_DIM, NULL);
        assert(vecs[i] != NULL);
        assert(vecs[i]->data != NULL);
        assert(((uintptr_t)vecs[i]->data % 64) == 0);
    }

    printf("[MEM TEST] Vector Data Alignment: PASSED\n");

    for (int i = 0; i < 100; i++) {
        pk_vector_free(vecs[i]);
    }
}

int main(void) {
    printf("========================================================================\n");
    printf(" POMAIKACHE QA: MODULE 3 - MEMORY SAFETY & 64-BYTE ALIGNMENT            \n");
    printf("========================================================================\n");
    test_64byte_alignment();
    test_vector_data_alignment();
    printf("[MODULE 3] ALL MEMORY ALIGNMENT TESTS PASSED.\n\n");
    return 0;
}
