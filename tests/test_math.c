/* ----------------------------------------------------------------------------
 * Pomaikache QA Suite Module 1: Functional Correctness & Math Verification
 * ---------------------------------------------------------------------------*/

#include "../src/pomaikache.h"
#include <assert.h>
#include <math.h>

// Scalar reference implementation for Dot Product
static float scalar_dot_product(const float *a, const float *b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Test 1: AVX2 Dot Product vs Scalar Reference across Dims (64, 128, 384, 768, 1536)
static void test_simd_vs_scalar_precision(void) {
    uint32_t dims[] = {64, 128, 384, 768, 1536};
    size_t num_dims = sizeof(dims) / sizeof(dims[0]);

    for (size_t d = 0; d < num_dims; d++) {
        uint32_t dim = dims[d];
        float *a = (float *)pk_malloc_aligned(dim * sizeof(float));
        float *b = (float *)pk_malloc_aligned(dim * sizeof(float));

        for (uint32_t i = 0; i < dim; i++) {
            a[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            b[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }

        float scalar_res = scalar_dot_product(a, b, dim);
        float simd_res = pmk_dot_product_avx2(a, b, dim);

        float delta = fabsf(scalar_res - simd_res);
        printf("[MATH TEST] Dim %4u: Scalar=%.6f, AVX2=%.6f, Delta=%.8f -> ", dim, scalar_res, simd_res, delta);
        assert(delta < 1e-4f);
        printf("PASSED\n");

        pk_free(a);
        pk_free(b);
    }
}

// Test 2: Mathematical Properties (Orthogonal & Identical Vectors)
static void test_orthogonal_identical_vectors(void) {
    uint32_t dim = 128;
    pk_vector_t *v1 = pk_vector_create(1, dim, NULL);
    pk_vector_t *v2 = pk_vector_create(2, dim, NULL);

    // Set identical non-zero values
    for (uint32_t i = 0; i < dim; i++) {
        v1->data[i] = 1.0f;
        v2->data[i] = 1.0f;
    }
    v1->norm = pk_vector_calc_norm(v1->data, dim);
    v2->norm = pk_vector_calc_norm(v2->data, dim);

    float cos_identical = pk_dist_cosine(v1, v2);
    printf("[MATH TEST] Identical Vectors Cosine Dist: %.6f -> ", cos_identical);
    assert(fabsf(cos_identical) < 1e-5f);
    printf("PASSED\n");

    // Set orthogonal vectors (v1 = [1,0,1,0...], v2 = [0,1,0,1...])
    for (uint32_t i = 0; i < dim; i++) {
        v1->data[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        v2->data[i] = (i % 2 == 1) ? 1.0f : 0.0f;
    }
    v1->norm = pk_vector_calc_norm(v1->data, dim);
    v2->norm = pk_vector_calc_norm(v2->data, dim);

    float cos_orthogonal = pk_dist_cosine(v1, v2);
    printf("[MATH TEST] Orthogonal Vectors Cosine Dist: %.6f -> ", cos_orthogonal);
    assert(fabsf(cos_orthogonal - 1.0f) < 1e-5f);
    printf("PASSED\n");

    pk_vector_free(v1);
    pk_vector_free(v2);
}

// Test 3: Zero Vectors and Edge Cases
static void test_edge_case_vectors(void) {
    uint32_t dim = 128;
    pk_vector_t *zero_vec = pk_vector_create(1, dim, NULL);
    assert(zero_vec->norm == 0.0f);

    pk_vector_t *norm_vec = pk_vector_create(2, dim, NULL);
    for (uint32_t i = 0; i < dim; i++) norm_vec->data[i] = 0.5f;
    norm_vec->norm = pk_vector_calc_norm(norm_vec->data, dim);

    float dist_zero = pk_dist_cosine(zero_vec, norm_vec);
    printf("[MATH TEST] Zero Vector Cosine Distance: %.6f -> ", dist_zero);
    assert(dist_zero == 1.0f);
    printf("PASSED\n");

    pk_vector_free(zero_vec);
    pk_vector_free(norm_vec);
}

int main(void) {
    printf("========================================================================\n");
    printf(" POMAIKACHE QA: MODULE 1 - MATH VERIFICATION & PRECISION                \n");
    printf("========================================================================\n");
    test_simd_vs_scalar_precision();
    test_orthogonal_identical_vectors();
    test_edge_case_vectors();
    printf("[MODULE 1] ALL MATH VERIFICATION TESTS PASSED.\n\n");
    return 0;
}
