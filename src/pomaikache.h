/* ----------------------------------------------------------------------------
 * Pomaikache - High-Performance Vector Caching & Retrieval Engine in C
 * 
 * Powered by:
 *   - palloc   : High-performance memory allocator (Microsoft mimalloc variant)
 *   - liburing : Linux io_uring asynchronous I/O framework
 * ---------------------------------------------------------------------------*/

#ifndef POMAIKACHE_H
#define POMAIKACHE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

#ifndef BLOCK_URING_CMD_DISCARD
#define BLOCK_URING_CMD_DISCARD 0
#endif
#ifndef BLOCK_URING_CMD_ZONE_RESET_ALL
#define BLOCK_URING_CMD_ZONE_RESET_ALL 1
#endif

#include "../palloc/include/palloc.h"
#include "../liburing/src/include/liburing.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------------
 * Memory Allocation Wrappers (STRICTLY USE palloc)
 * ---------------------------------------------------------------------------*/
#define pk_malloc(size)                 pa_malloc(size)
#define pk_calloc(count, size)          pa_calloc(count, size)
#define pk_realloc(ptr, size)           pa_realloc(ptr, size)
#define pk_free(ptr)                    pa_free(ptr)
#define pk_strdup(s)                    pa_strdup(s)

// 64-byte alignment for AVX2 / AVX-512 SIMD vector operations
#define PK_ALIGNMENT                    64
#define pk_malloc_aligned(size)         pa_malloc_aligned(size, PK_ALIGNMENT)
#define pk_zalloc_aligned(size)         pa_zalloc_aligned(size, PK_ALIGNMENT)

// Thread-local heap abstractions
typedef pa_heap_t                       pk_heap_t;
#define pk_heap_new()                   pa_heap_new()
#define pk_heap_delete(hp)              pa_heap_delete(hp)
#define pk_heap_malloc(hp, size)        pa_heap_malloc(hp, size)
#define pk_heap_zalloc(hp, size)        pa_heap_zalloc(hp, size)
#define pk_heap_free(hp, ptr)           pa_free(ptr)

/* ----------------------------------------------------------------------------
 * Vector Math & Metric Definitions
 * ---------------------------------------------------------------------------*/
typedef enum {
    PK_METRIC_COSINE = 0,
    PK_METRIC_L2     = 1,
    PK_METRIC_DOT    = 2
} pk_metric_type_t;

// Vector Payload Structure
typedef struct {
    uint64_t id;            // Unique vector identifier
    uint32_t dim;           // Vector dimension
    float norm;             // Pre-calculated L2 norm (magnitude)
    float *data;            // 64-byte aligned float array allocated via pk_malloc_aligned
} pk_vector_t;

// Nearest Neighbor Search Result Item
typedef struct {
    uint64_t id;            // Vector ID
    float score;            // Distance or similarity score
} pk_search_result_t;

/* ----------------------------------------------------------------------------
 * L1 LRU Vector Cache Structs
 * ---------------------------------------------------------------------------*/
#define PK_LRU_HASH_SIZE 1024

typedef struct pk_lru_entry {
    char key[64];
    pk_vector_t *vec;
    pk_search_result_t *cached_results;
    uint32_t cached_count;
    uint64_t expires_at_ms;     // Expiration timestamp in ms (0 = infinite)
    struct pk_lru_entry *prev;
    struct pk_lru_entry *next;
    struct pk_lru_entry *hash_next;
} pk_lru_entry_t;

typedef struct {
    size_t capacity;
    size_t size;
    uint64_t hits;
    uint64_t misses;
    uint64_t expired_count;
    uint64_t default_ttl_ms;    // Default TTL in milliseconds (0 = no TTL)
    pk_lru_entry_t *head;
    pk_lru_entry_t *tail;
    pk_lru_entry_t *buckets[PK_LRU_HASH_SIZE];
    pthread_mutex_t lock;
} pk_lru_cache_t;

/* ----------------------------------------------------------------------------
 * L2 Vector Index (SIMD Flat Scan + HNSW Index)
 * ---------------------------------------------------------------------------*/
#define PK_HNSW_MAX_LEVELS 16
#define PK_HNSW_DEFAULT_M  16
#define PK_HNSW_DEFAULT_EF 64

typedef struct pk_hnsw_node {
    uint64_t id;
    pk_vector_t *vector;
    uint32_t level;
    struct pk_hnsw_node ***neighbors;  // neighbors[level][i]
    uint32_t *neighbor_count;           // neighbor_count[level]
} pk_hnsw_node_t;

typedef struct {
    uint32_t dim;
    pk_metric_type_t metric;
    
    // Flat storage array of vectors (always updated for 100% accuracy SIMD scan)
    pk_vector_t **vectors;
    size_t count;
    size_t capacity;

    // HNSW Graph Index
    bool use_hnsw;
    uint32_t m;
    uint32_t ef_construction;
    uint32_t ef_search;
    int max_level;
    pk_hnsw_node_t *entry_point;
    pk_hnsw_node_t **nodes;
    size_t node_count;

    pthread_rwlock_t rwlock;
} pk_index_t;

/* ----------------------------------------------------------------------------
 * Engine State & Networking (liburing)
 * ---------------------------------------------------------------------------*/
#define PK_QUEUE_DEPTH 256
#define PK_MAX_BUF_SIZE 4096

typedef struct {
    uint16_t port;
    uint32_t vector_dim;
    size_t lru_capacity;
    pk_metric_type_t metric;
    char wal_path[256];
    char snapshot_path[256];
} pk_config_t;

typedef struct {
    pk_config_t config;
    pk_index_t *index;
    pk_lru_cache_t *cache;
    struct io_uring ring;
    int server_fd;
    int wal_fd;
    bool running;
    pk_heap_t *heap;
} pk_engine_t;

/* ----------------------------------------------------------------------------
 * Public API Prototypes
 * ---------------------------------------------------------------------------*/

// Vector Allocation & Math
pk_vector_t *pk_vector_create(uint64_t id, uint32_t dim, const float *data);
void pk_vector_free(pk_vector_t *vec);
float pk_vector_calc_norm(const float *data, uint32_t dim);

// SIMD Distance Functions
float pk_dist_cosine(const pk_vector_t *a, const pk_vector_t *b);
float pk_dist_l2(const pk_vector_t *a, const pk_vector_t *b);
float pk_dist_dot(const pk_vector_t *a, const pk_vector_t *b);
float pk_vector_distance(const pk_vector_t *a, const pk_vector_t *b, pk_metric_type_t metric);

// L1 LRU Cache
pk_lru_cache_t *pk_lru_create(size_t capacity);
void pk_lru_free(pk_lru_cache_t *cache);
pk_vector_t *pk_lru_get(pk_lru_cache_t *cache, const char *key);
void pk_lru_put(pk_lru_cache_t *cache, const char *key, pk_vector_t *vec);
void pk_lru_put_ttl(pk_lru_cache_t *cache, const char *key, pk_vector_t *vec, uint64_t ttl_ms);
bool pk_lru_del(pk_lru_cache_t *cache, const char *key);
size_t pk_lru_purge_expired(pk_lru_cache_t *cache);

// L2 Vector Index
pk_index_t *pk_index_create(uint32_t dim, pk_metric_type_t metric, bool use_hnsw);
void pk_index_free(pk_index_t *index);
int pk_index_insert(pk_index_t *index, pk_vector_t *vec);
bool pk_index_remove(pk_index_t *index, uint64_t id);
size_t pk_index_search(pk_index_t *index, const pk_vector_t *query, uint32_t k, pk_search_result_t *results);

/* ----------------------------------------------------------------------------
 * Immutable Append-Only + Circular Overwrite Vector Ring Buffer
 * ---------------------------------------------------------------------------*/
typedef struct {
    uint32_t dim;
    size_t capacity;            // Max vector slots
    size_t head;                // Ring insertion head
    size_t count;               // Active vectors count
    uint64_t total_appended;    // Monotonically increasing vector count
    
    // Contiguous 64-byte aligned memory allocations using palloc
    float *data_ring;           // Contiguous float buffer [capacity * dim]
    uint64_t *ids_ring;         // Vector IDs array [capacity]
    float *norms_ring;          // Pre-computed vector norms [capacity]
    
    pk_metric_type_t metric;
    pthread_rwlock_t rwlock;
} pk_circular_ring_t;

pk_circular_ring_t *pk_ring_create(size_t capacity, uint32_t dim, pk_metric_type_t metric);
void pk_ring_free(pk_circular_ring_t *ring);
size_t pk_ring_append(pk_circular_ring_t *ring, uint64_t id, const float *data);
size_t pk_ring_search(pk_circular_ring_t *ring, const float *query_data, uint32_t k, pk_search_result_t *results);

/* ----------------------------------------------------------------------------
 * Fixed-Size Top-K Min-Heap Structure (Early Rejection Thresholding)
 * ---------------------------------------------------------------------------*/
typedef struct {
    uint64_t id;
    float score;
} pmk_heap_node_t;

typedef struct {
    uint32_t k;
    uint32_t size;
    pmk_heap_node_t *nodes; // Fixed size array of k elements allocated via palloc
} pmk_topk_heap_t;

pmk_topk_heap_t *pmk_topk_heap_create(uint32_t k);
void pmk_topk_heap_free(pmk_topk_heap_t *heap);
void pmk_topk_heap_reset(pmk_topk_heap_t *heap);
bool pmk_topk_heap_push(pmk_topk_heap_t *heap, uint64_t id, float score);
size_t pmk_topk_heap_extract(pmk_topk_heap_t *heap, pk_search_result_t *results);

/* ----------------------------------------------------------------------------
 * 4x Unrolled AVX2 FMA Dot Product SIMD Kernel
 * ---------------------------------------------------------------------------*/
float pmk_dot_product_avx2(const float *a, const float *b, uint32_t dim);

/* ----------------------------------------------------------------------------
 * Canonical IVF-Flat Index (FAISS Architecture)
 * ---------------------------------------------------------------------------*/
typedef struct {
    uint64_t id;
    float *data; // 64-byte aligned vector payload allocated via palloc
} pmk_ivf_list_node_t;

typedef struct {
    size_t count;
    size_t capacity;
    pmk_ivf_list_node_t *entries;
} pmk_ivf_posting_list_t;

typedef struct {
    uint32_t dim;
    uint32_t nlist;
    uint32_t nprobe;
    float *centroids; // Flat array [nlist * dim], 64-byte aligned via palloc
    pmk_ivf_posting_list_t *lists; // Posting lists array [nlist]
    pthread_rwlock_t rwlock;
} pmk_ivf_index_t;

pmk_ivf_index_t *pmk_ivf_init(uint32_t dim, uint32_t nlist, uint32_t nprobe);
void pmk_ivf_free(pmk_ivf_index_t *ivf);
int pmk_ivf_train_centroids(pmk_ivf_index_t *ivf, const float *training_data, size_t num_vectors);
int pmk_ivf_add(pmk_ivf_index_t *ivf, uint64_t id, const float *vector_data);
size_t pmk_ivf_search(pmk_ivf_index_t *ivf, const float *query_data, uint32_t k, pk_search_result_t *results);

/* ----------------------------------------------------------------------------
 * Canonical HNSW Index (hnswlib Architecture - Malkov & Yashunin 2018)
 * ---------------------------------------------------------------------------*/
typedef struct pmk_hnsw_element {
    uint64_t id;
    float *data; // 64-byte aligned vector payload allocated via palloc
    int level;
    uint32_t **neighbors; // Multi-layer neighbor lists: neighbors[l][0] = count
} pmk_hnsw_element_t;

typedef struct {
    uint32_t dim;
    uint32_t max_elements;
    uint32_t cur_element_count;
    uint32_t M;
    uint32_t M0; // 2 * M (for layer 0)
    uint32_t ef_construction;
    uint32_t ef_search;
    int max_level;
    int enter_node_id; // Entry point element index (-1 if empty)
    double mult; // Level generation multiplier = 1.0 / log(M)
    
    pmk_hnsw_element_t **elements; // Flat contiguous elements lookup table
    pthread_rwlock_t rwlock;
} pmk_hnsw_index_t;

pmk_hnsw_index_t *pmk_hnsw_init(uint32_t dim, uint32_t max_elements, uint32_t M, uint32_t ef_construction, uint32_t ef_search);
void pmk_hnsw_free(pmk_hnsw_index_t *hnsw);
int pmk_hnsw_add(pmk_hnsw_index_t *hnsw, uint64_t id, const float *vector_data);
size_t pmk_hnsw_search_knn(pmk_hnsw_index_t *hnsw, const float *query_data, uint32_t k, pk_search_result_t *results);

// Engine & Persistence (liburing)
pk_engine_t *pk_engine_create(const pk_config_t *config);
void pk_engine_free(pk_engine_t *engine);
int pk_wal_init(pk_engine_t *engine);
int pk_wal_append(pk_engine_t *engine, const pk_vector_t *vec);
int pk_wal_recover(pk_engine_t *engine);
int pk_snapshot_save(pk_engine_t *engine, const char *filepath);
int pk_snapshot_load(pk_engine_t *engine, const char *filepath);
int pk_engine_start_server(pk_engine_t *engine);

#ifdef __cplusplus
}
#endif

#endif // POMAIKACHE_H
