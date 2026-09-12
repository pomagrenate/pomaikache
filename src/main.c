/* ----------------------------------------------------------------------------
 * Pomaikache - High-Performance Vector Caching & Retrieval Engine in C
 * Main CLI Entry Point & SIMD Benchmark Suite
 * ---------------------------------------------------------------------------*/

#include "pomaikache.h"
#include <time.h>
#include <getopt.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void print_banner(void) {
    printf("========================================================================\n");
    printf("                  POMAIKACHE - Vector Caching Engine                    \n");
    printf("            Powered by palloc Allocator & liburing (io_uring)          \n");
    printf("========================================================================\n\n");
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --port <port>       TCP server port (default: 9090)\n");
    printf("  --dim <dim>         Vector dimension (default: 128)\n");
    printf("  --capacity <n>      LRU cache capacity (default: 10000)\n");
    printf("  --bench             Run high-speed SIMD vector benchmark\n");
    printf("  --help              Display this help message\n");
}

static void run_benchmark(uint32_t dim, size_t num_vectors, size_t num_queries) {
    print_banner();
    printf("[BENCHMARK] Initializing benchmark with %zu vectors (dim=%u), %zu queries...\n\n",
           num_vectors, dim, num_queries);
           
    // Create Index & LRU Cache
    pk_index_t *index = pk_index_create(dim, PK_METRIC_COSINE, false);
    pk_lru_cache_t *cache = pk_lru_create(1000);
    
    // Allocate temporary float buffer using palloc
    float *raw_data = (float *)pk_malloc(dim * sizeof(float));
    
    printf("1. Population Phase: Creating & Inserting %zu vectors via palloc...\n", num_vectors);
    double start_pop = get_time_sec();
    for (size_t i = 0; i < num_vectors; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            raw_data[d] = (float)rand() / (float)RAND_MAX;
        }
        pk_vector_t *vec = pk_vector_create((uint64_t)(i + 1), dim, raw_data);
        pk_index_insert(index, vec);
    }
    double elapsed_pop = get_time_sec() - start_pop;
    printf("   -> Populated %zu vectors in %.4f seconds (%.2f ops/sec)\n\n",
           num_vectors, elapsed_pop, num_vectors / elapsed_pop);
           
    // Create query vector
    for (uint32_t d = 0; d < dim; d++) {
        raw_data[d] = (float)rand() / (float)RAND_MAX;
    }
    pk_vector_t *query = pk_vector_create(999999, dim, raw_data);
    
    pk_search_result_t results[10];
    
    printf("2. SIMD Search Phase: Executing %zu Cosine distance queries...\n", num_queries);
    double start_search = get_time_sec();
    for (size_t q = 0; q < num_queries; q++) {
        pk_index_search(index, query, 10, results);
    }
    double elapsed_search = get_time_sec() - start_search;
    double qps = num_queries / elapsed_search;
    double avg_lat_us = (elapsed_search / num_queries) * 1e6;
    
    printf("   -> Total Time    : %.4f seconds\n", elapsed_search);
    printf("   -> Throughput    : %.2f Queries/Sec (QPS)\n", qps);
    printf("   -> Avg Latency   : %.2f us per query\n\n", avg_lat_us);
    
    printf("3. L1 Cache Throughput & TTL Phase...\n");
    double start_cache = get_time_sec();
    for (size_t c = 0; c < num_queries; c++) {
        char key[32];
        snprintf(key, sizeof(key), "key_%zu", c % 500);
        
        pk_vector_t *cached = pk_lru_get(cache, key);
        if (!cached) {
            pk_vector_t *new_vec = pk_vector_create(c, dim, raw_data);
            // Put with 100ms TTL for testing
            pk_lru_put_ttl(cache, key, new_vec, (c % 10 == 0) ? 50 : 0);
        }
    }
    double elapsed_cache = get_time_sec() - start_cache;
    printf("   -> Cache Ops Time : %.4f seconds\n", elapsed_cache);
    printf("   -> Cache Hits     : %lu\n", cache->hits);
    printf("   -> Cache Misses   : %lu\n", cache->misses);
    printf("   -> Expired Purges : %lu\n", cache->expired_count);
    printf("   -> Hit Rate       : %.2f%%\n\n", (double)cache->hits / (cache->hits + cache->misses) * 100.0);
    
    printf("4. Circular Overwrite Ring Buffer Phase...\n");
    pk_circular_ring_t *ring = pk_ring_create(1000, dim, PK_METRIC_COSINE);
    double start_ring = get_time_sec();
    for (size_t r = 0; r < 2500; r++) { // Test 2.5x capacity (causes 1.5x wrap-around overwrite)
        pk_ring_append(ring, (uint64_t)r, raw_data);
    }
    double elapsed_ring = get_time_sec() - start_ring;
    printf("   -> Ring Appended  : 2500 items into 1000 slot ring\n");
    printf("   -> Active Count   : %zu vectors\n", ring->count);
    printf("   -> Total Appended : %lu vectors\n", ring->total_appended);
    printf("   -> Ring Ops Time  : %.4f seconds (%.2f ops/sec)\n\n", elapsed_ring, 2500.0 / elapsed_ring);
    pk_ring_free(ring);
    
    printf("5. FAISS IVF-Flat & HNSW Index Verification...\n");
    pmk_ivf_index_t *ivf = pmk_ivf_init(dim, 16, 4);
    pmk_ivf_train_centroids(ivf, raw_data, 16);
    
    pmk_hnsw_index_t *hnsw = pmk_hnsw_init(dim, 1000, 16, 64, 64);
    
    for (size_t i = 0; i < 500; i++) {
        pmk_ivf_add(ivf, (uint64_t)(i + 1), raw_data);
        pmk_hnsw_add(hnsw, (uint64_t)(i + 1), raw_data);
    }
    
    pk_search_result_t ivf_results[5];
    pk_search_result_t hnsw_results[5];
    size_t ivf_found = pmk_ivf_search(ivf, raw_data, 5, ivf_results);
    size_t hnsw_found = pmk_hnsw_search_knn(hnsw, raw_data, 5, hnsw_results);
    
    printf("   -> IVF-Flat Found : %zu top-5 nearest neighbors\n", ivf_found);
    printf("   -> HNSW Found     : %zu top-5 nearest neighbors\n\n", hnsw_found);
    
    pmk_ivf_free(ivf);
    pmk_hnsw_free(hnsw);
    
    printf("6. Memory Allocation Statistics (palloc):\n");
    pa_stats_print(NULL);
    printf("\n");
    
    // Cleanup using palloc free wrappers
    pk_vector_free(query);
    pk_free(raw_data);
    pk_index_free(index);
    pk_lru_free(cache);
    
    printf("[BENCHMARK] Completed successfully.\n");
}

int main(int argc, char **argv) {
    uint16_t port = 9090;
    uint32_t dim = 128;
    size_t capacity = 10000;
    bool bench_mode = false;
    
    static struct option long_options[] = {
        {"port",     required_argument, 0, 'p'},
        {"dim",      required_argument, 0, 'd'},
        {"capacity", required_argument, 0, 'c'},
        {"bench",    no_argument,       0, 'b'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "p:d:c:bh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'p': port = (uint16_t)atoi(optarg); break;
            case 'd': dim = (uint32_t)atoi(optarg); break;
            case 'c': capacity = (size_t)atol(optarg); break;
            case 'b': bench_mode = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }
    
    if (bench_mode) {
        run_benchmark(dim, 10000, 5000);
        return 0;
    }
    
    print_banner();
    
    pk_config_t config;
    memset(&config, 0, sizeof(config));
    config.port = port;
    config.vector_dim = dim;
    config.lru_capacity = capacity;
    config.metric = PK_METRIC_COSINE;
    strncpy(config.wal_path, "pomaikache.wal", sizeof(config.wal_path));
    strncpy(config.snapshot_path, "pomaikache.snap", sizeof(config.snapshot_path));
    
    pk_engine_t *engine = pk_engine_create(&config);
    if (!engine) {
        fprintf(stderr, "Failed to create pomaikache engine\n");
        return 1;
    }
    
    pk_wal_init(engine);
    pk_wal_recover(engine);
    
    int res = pk_engine_start_server(engine);
    
    pk_engine_free(engine);
    return res;
}
