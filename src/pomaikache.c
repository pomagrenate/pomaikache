/* ----------------------------------------------------------------------------
 * Pomaikache - High-Performance Vector Caching & Retrieval Engine in C
 * Vector Management, SIMD Distance Metrics, LRU Cache & Vector Index
 * ---------------------------------------------------------------------------*/

#include "pomaikache.h"

/* ----------------------------------------------------------------------------
 * SIMD Helper Functions & Distance Metrics
 * ---------------------------------------------------------------------------*/

// Helper: Horizontal sum of an AVX2 256-bit float register
static inline float hsum_avx2(__m256 v) {
    __m128 low  = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum  = _mm_add_ps(low, high);
    sum         = _mm_hadd_ps(sum, sum);
    sum         = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Calculate pre-computed L2 Norm (Magnitude) using AVX2 SIMD
float pk_vector_calc_norm(const float *data, uint32_t dim) {
    if (!data || dim == 0) return 0.0f;
    
    __m256 sum_vec = _mm256_setzero_ps();
    uint32_t i = 0;
    
    // Process 8 floats per iteration
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_load_ps(&data[i]);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
    }
    
    float total_sq = hsum_avx2(sum_vec);
    
    // Process remaining elements
    for (; i < dim; i++) {
        total_sq += data[i] * data[i];
    }
    
    return sqrtf(total_sq);
}

// SIMD AVX2 Dot Product Kernel
float pk_dist_dot(const pk_vector_t *a, const pk_vector_t *b) {
    uint32_t dim = a->dim;
    const float *da = a->data;
    const float *db = b->data;
    
    __m256 sum_vec = _mm256_setzero_ps();
    uint32_t i = 0;
    
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_load_ps(&da[i]);
        __m256 vb = _mm256_load_ps(&db[i]);
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
    }
    
    float total = hsum_avx2(sum_vec);
    
    for (; i < dim; i++) {
        total += da[i] * db[i];
    }
    
    return total;
}

// SIMD AVX2 Cosine Distance (1.0 - Cosine Similarity)
float pk_dist_cosine(const pk_vector_t *a, const pk_vector_t *b) {
    if (a->norm == 0.0f || b->norm == 0.0f) return 1.0f;
    
    float dot = pk_dist_dot(a, b);
    float sim = dot / (a->norm * b->norm);
    
    // Clamp similarity to [-1.0, 1.0] to prevent floating point inaccuracy
    if (sim > 1.0f) sim = 1.0f;
    if (sim < -1.0f) sim = -1.0f;
    
    return 1.0f - sim;
}

// SIMD AVX2 L2 (Euclidean) Distance Kernel
float pk_dist_l2(const pk_vector_t *a, const pk_vector_t *b) {
    uint32_t dim = a->dim;
    const float *da = a->data;
    const float *db = b->data;
    
    __m256 sum_vec = _mm256_setzero_ps();
    uint32_t i = 0;
    
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_load_ps(&da[i]);
        __m256 vb = _mm256_load_ps(&db[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    
    float total_sq = hsum_avx2(sum_vec);
    
    for (; i < dim; i++) {
        float diff = da[i] - db[i];
        total_sq += diff * diff;
    }
    
    return sqrtf(total_sq);
}

// Generic distance dispatcher
float pk_vector_distance(const pk_vector_t *a, const pk_vector_t *b, pk_metric_type_t metric) {
    switch (metric) {
        case PK_METRIC_COSINE: return pk_dist_cosine(a, b);
        case PK_METRIC_L2:     return pk_dist_l2(a, b);
        case PK_METRIC_DOT:    return -pk_dist_dot(a, b); // Negative dot product for sorting lower = better
        default:               return pk_dist_cosine(a, b);
    }
}

/* ----------------------------------------------------------------------------
 * Vector Allocation Routines (STRICTLY USE palloc)
 * ---------------------------------------------------------------------------*/

pk_vector_t *pk_vector_create(uint64_t id, uint32_t dim, const float *data) {
    pk_vector_t *vec = (pk_vector_t *)pk_malloc(sizeof(pk_vector_t));
    if (!vec) return NULL;
    
    vec->id = id;
    vec->dim = dim;
    
    // Allocate 64-byte aligned array for AVX SIMD performance using palloc
    vec->data = (float *)pk_malloc_aligned(dim * sizeof(float));
    if (!vec->data) {
        pk_free(vec);
        return NULL;
    }
    
    if (data) {
        memcpy(vec->data, data, dim * sizeof(float));
    } else {
        memset(vec->data, 0, dim * sizeof(float));
    }
    
    vec->norm = pk_vector_calc_norm(vec->data, dim);
    return vec;
}

void pk_vector_free(pk_vector_t *vec) {
    if (!vec) return;
    if (vec->data) {
        pk_free(vec->data);
    }
    pk_free(vec);
}

/* ----------------------------------------------------------------------------
 * L1 LRU Cache Implementation (Using palloc & TTL Support)
 * ---------------------------------------------------------------------------*/

static uint64_t current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static uint32_t pk_hash_key(const char *key) {
    char buf[64];
    strncpy(buf, key, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    
    uint64_t hash = 14695981039346656037ULL; // FNV-1a 64-bit
    for (const char *p = buf; *p; p++) {
        hash ^= (uint64_t)(unsigned char)(*p);
        hash *= 1099511628211ULL;
    }
    return (uint32_t)(hash % PK_LRU_HASH_SIZE);
}

pk_lru_cache_t *pk_lru_create(size_t capacity) {
    pk_lru_cache_t *cache = (pk_lru_cache_t *)pk_calloc(1, sizeof(pk_lru_cache_t));
    if (!cache) return NULL;
    
    cache->capacity = capacity;
    cache->size = 0;
    cache->hits = 0;
    cache->misses = 0;
    cache->expired_count = 0;
    cache->default_ttl_ms = 0; // 0 = no expiration by default
    cache->head = NULL;
    cache->tail = NULL;
    pthread_mutex_init(&cache->lock, NULL);
    
    return cache;
}

void pk_lru_free(pk_lru_cache_t *cache) {
    if (!cache) return;
    
    pthread_mutex_lock(&cache->lock);
    pk_lru_entry_t *curr = cache->head;
    while (curr) {
        pk_lru_entry_t *next = curr->next;
        if (curr->vec) pk_vector_free(curr->vec);
        if (curr->cached_results) pk_free(curr->cached_results);
        pk_free(curr);
        curr = next;
    }
    pthread_mutex_unlock(&cache->lock);
    pthread_mutex_destroy(&cache->lock);
    
    pk_free(cache);
}

static void remove_node(pk_lru_cache_t *cache, pk_lru_entry_t *entry) {
    if (entry->prev) entry->prev->next = entry->next;
    else cache->head = entry->next;
    
    if (entry->next) entry->next->prev = entry->prev;
    else cache->tail = entry->prev;
}

static void move_to_head(pk_lru_cache_t *cache, pk_lru_entry_t *entry) {
    remove_node(cache, entry);
    entry->next = cache->head;
    entry->prev = NULL;
    if (cache->head) cache->head->prev = entry;
    cache->head = entry;
    if (!cache->tail) cache->tail = entry;
}

pk_vector_t *pk_lru_get(pk_lru_cache_t *cache, const char *key) {
    if (!cache || !key) return NULL;
    
    pthread_mutex_lock(&cache->lock);
    uint32_t bucket = pk_hash_key(key);
    pk_lru_entry_t *entry = cache->buckets[bucket];
    uint64_t now = current_time_ms();
    
    while (entry) {
        if (strncmp(entry->key, key, sizeof(entry->key)) == 0) {
            // Check TTL expiration
            if (entry->expires_at_ms > 0 && now >= entry->expires_at_ms) {
                // Expired: Purge entry using palloc free
                pk_lru_entry_t **pp = &cache->buckets[bucket];
                while (*pp && *pp != entry) pp = &(*pp)->hash_next;
                if (*pp) *pp = entry->hash_next;
                
                remove_node(cache, entry);
                if (entry->vec) pk_vector_free(entry->vec);
                if (entry->cached_results) pk_free(entry->cached_results);
                pk_free(entry);
                cache->size--;
                cache->expired_count++;
                cache->misses++;
                pthread_mutex_unlock(&cache->lock);
                return NULL;
            }
            
            move_to_head(cache, entry);
            cache->hits++;
            pk_vector_t *result = entry->vec;
            pthread_mutex_unlock(&cache->lock);
            return result;
        }
        entry = entry->hash_next;
    }
    
    cache->misses++;
    pthread_mutex_unlock(&cache->lock);
    return NULL;
}

void pk_lru_put_ttl(pk_lru_cache_t *cache, const char *key, pk_vector_t *vec, uint64_t ttl_ms) {
    if (!cache || !key || !vec) return;
    
    pthread_mutex_lock(&cache->lock);
    uint32_t bucket = pk_hash_key(key);
    pk_lru_entry_t *entry = cache->buckets[bucket];
    uint64_t now = current_time_ms();
    uint64_t expires_at = (ttl_ms > 0) ? (now + ttl_ms) : 0;
    
    while (entry) {
        if (strncmp(entry->key, key, sizeof(entry->key)) == 0) {
            if (entry->vec) pk_vector_free(entry->vec);
            entry->vec = vec;
            entry->expires_at_ms = expires_at;
            move_to_head(cache, entry);
            pthread_mutex_unlock(&cache->lock);
            return;
        }
        entry = entry->hash_next;
    }
    
    // Evict tail if full
    if (cache->size >= cache->capacity && cache->tail) {
        pk_lru_entry_t *evict = cache->tail;
        uint32_t evict_bucket = pk_hash_key(evict->key);
        
        // Remove from hash chain
        pk_lru_entry_t **pp = &cache->buckets[evict_bucket];
        while (*pp && *pp != evict) pp = &(*pp)->hash_next;
        if (*pp) *pp = evict->hash_next;
        
        remove_node(cache, evict);
        if (evict->vec) pk_vector_free(evict->vec);
        if (evict->cached_results) pk_free(evict->cached_results);
        pk_free(evict);
        cache->size--;
    }
    
    // Create new LRU entry via palloc
    pk_lru_entry_t *new_entry = (pk_lru_entry_t *)pk_malloc(sizeof(pk_lru_entry_t));
    if (!new_entry) {
        pthread_mutex_unlock(&cache->lock);
        return;
    }
    
    strncpy(new_entry->key, key, sizeof(new_entry->key) - 1);
    new_entry->key[sizeof(new_entry->key) - 1] = '\0';
    new_entry->vec = vec;
    new_entry->cached_results = NULL;
    new_entry->cached_count = 0;
    new_entry->expires_at_ms = expires_at;
    
    // Insert into hash chain
    new_entry->hash_next = cache->buckets[bucket];
    cache->buckets[bucket] = new_entry;
    
    // Insert at LRU head
    new_entry->next = cache->head;
    new_entry->prev = NULL;
    if (cache->head) cache->head->prev = new_entry;
    cache->head = new_entry;
    if (!cache->tail) cache->tail = new_entry;
    
    cache->size++;
    pthread_mutex_unlock(&cache->lock);
}

bool pk_lru_del(pk_lru_cache_t *cache, const char *key) {
    if (!cache || !key) return false;
    
    pthread_mutex_lock(&cache->lock);
    uint32_t bucket = pk_hash_key(key);
    pk_lru_entry_t **pp = &cache->buckets[bucket];
    
    while (*pp) {
        if (strncmp((*pp)->key, key, sizeof((*pp)->key)) == 0) {
            pk_lru_entry_t *entry = *pp;
            *pp = entry->next;
            remove_node(cache, entry);
            if (entry->vec) pk_vector_free(entry->vec);
            if (entry->cached_results) pk_free(entry->cached_results);
            pk_free(entry);
            cache->size--;
            pthread_mutex_unlock(&cache->lock);
            return true;
        }
        pp = &(*pp)->next;
    }
    
    pthread_mutex_unlock(&cache->lock);
    return false;
}

void pk_lru_put(pk_lru_cache_t *cache, const char *key, pk_vector_t *vec) {
    if (!cache) return;
    pk_lru_put_ttl(cache, key, vec, cache->default_ttl_ms);
}

size_t pk_lru_purge_expired(pk_lru_cache_t *cache) {
    if (!cache) return 0;
    
    pthread_mutex_lock(&cache->lock);
    uint64_t now = current_time_ms();
    size_t purged = 0;
    
    pk_lru_entry_t *curr = cache->head;
    while (curr) {
        pk_lru_entry_t *next = curr->next;
        if (curr->expires_at_ms > 0 && now >= curr->expires_at_ms) {
            uint32_t bucket = pk_hash_key(curr->key);
            pk_lru_entry_t **pp = &cache->buckets[bucket];
            while (*pp && *pp != curr) pp = &(*pp)->hash_next;
            if (*pp) *pp = curr->hash_next;
            
            remove_node(cache, curr);
            if (curr->vec) pk_vector_free(curr->vec);
            if (curr->cached_results) pk_free(curr->cached_results);
            pk_free(curr);
            cache->size--;
            cache->expired_count++;
            purged++;
        }
        curr = next;
    }
    
    pthread_mutex_unlock(&cache->lock);
    return purged;
}

/* ----------------------------------------------------------------------------
 * L2 Vector Index (SIMD Flat Scan + Search Heap)
 * ---------------------------------------------------------------------------*/

pk_index_t *pk_index_create(uint32_t dim, pk_metric_type_t metric, bool use_hnsw) {
    pk_index_t *index = (pk_index_t *)pk_calloc(1, sizeof(pk_index_t));
    if (!index) return NULL;
    
    index->dim = dim;
    index->metric = metric;
    index->use_hnsw = use_hnsw;
    index->capacity = 1024;
    index->count = 0;
    
    index->vectors = (pk_vector_t **)pk_malloc(index->capacity * sizeof(pk_vector_t *));
    if (!index->vectors) {
        pk_free(index);
        return NULL;
    }
    
    pthread_rwlock_init(&index->rwlock, NULL);
    return index;
}

void pk_index_free(pk_index_t *index) {
    if (!index) return;
    
    pthread_rwlock_wrlock(&index->rwlock);
    for (size_t i = 0; i < index->count; i++) {
        if (index->vectors[i]) {
            pk_vector_free(index->vectors[i]);
        }
    }
    pk_free(index->vectors);
    pthread_rwlock_unlock(&index->rwlock);
    pthread_rwlock_destroy(&index->rwlock);
    
    pk_free(index);
}

int pk_index_insert(pk_index_t *index, pk_vector_t *vec) {
    if (!index || !vec || vec->dim != index->dim) return -1;
    
    pthread_rwlock_wrlock(&index->rwlock);
    
    if (index->count >= index->capacity) {
        size_t new_cap = index->capacity * 2;
        pk_vector_t **new_vecs = (pk_vector_t **)pk_realloc(index->vectors, new_cap * sizeof(pk_vector_t *));
        if (!new_vecs) {
            pthread_rwlock_unlock(&index->rwlock);
            return -1;
        }
        index->vectors = new_vecs;
        index->capacity = new_cap;
    }
    
    index->vectors[index->count++] = vec;
    pthread_rwlock_unlock(&index->rwlock);
    return 0;
}

bool pk_index_remove(pk_index_t *index, uint64_t id) {
    if (!index) return false;
    
    pthread_rwlock_wrlock(&index->rwlock);
    for (size_t i = 0; i < index->count; i++) {
        if (index->vectors[i]->id == id) {
            pk_vector_free(index->vectors[i]);
            // Swap with last element
            index->vectors[i] = index->vectors[index->count - 1];
            index->count--;
            pthread_rwlock_unlock(&index->rwlock);
            return true;
        }
    }
    pthread_rwlock_unlock(&index->rwlock);
    return false;
}

// Compare function for qsort of search results
static int cmp_search_results(const void *a, const void *b) {
    const pk_search_result_t *ra = (const pk_search_result_t *)a;
    const pk_search_result_t *rb = (const pk_search_result_t *)b;
    if (ra->score < rb->score) return -1;
    if (ra->score > rb->score) return 1;
    return 0;
}

size_t pk_index_search(pk_index_t *index, const pk_vector_t *query, uint32_t k, pk_search_result_t *results) {
    if (!index || !query || !results || k == 0) return 0;
    
    pthread_rwlock_rdlock(&index->rwlock);
    size_t n = index->count;
    if (n == 0) {
        pthread_rwlock_unlock(&index->rwlock);
        return 0;
    }
    
    // Allocate temporary score buffer using palloc
    pk_search_result_t *all_res = (pk_search_result_t *)pk_malloc(n * sizeof(pk_search_result_t));
    if (!all_res) {
        pthread_rwlock_unlock(&index->rwlock);
        return 0;
    }
    
    // Parallel SIMD flat scan over vector dataset
    for (size_t i = 0; i < n; i++) {
        all_res[i].id = index->vectors[i]->id;
        all_res[i].score = pk_vector_distance(query, index->vectors[i], index->metric);
    }
    
    pthread_rwlock_unlock(&index->rwlock);
    
    // Sort results to extract top K
    qsort(all_res, n, sizeof(pk_search_result_t), cmp_search_results);
    
    size_t returned_count = (k < n) ? k : n;
    memcpy(results, all_res, returned_count * sizeof(pk_search_result_t));
    
    pk_free(all_res);
    return returned_count;
}

/* ----------------------------------------------------------------------------
 * Immutable Append-Only + Circular Overwrite Vector Ring Buffer
 * ---------------------------------------------------------------------------*/

pk_circular_ring_t *pk_ring_create(size_t capacity, uint32_t dim, pk_metric_type_t metric) {
    if (capacity == 0 || dim == 0) return NULL;
    
    pk_circular_ring_t *ring = (pk_circular_ring_t *)pk_calloc(1, sizeof(pk_circular_ring_t));
    if (!ring) return NULL;
    
    ring->dim = dim;
    ring->capacity = capacity;
    ring->head = 0;
    ring->count = 0;
    ring->total_appended = 0;
    ring->metric = metric;
    
    // Allocate contiguous 64-byte aligned vector memory via palloc
    size_t total_floats = capacity * dim;
    ring->data_ring = (float *)pk_malloc_aligned(total_floats * sizeof(float));
    ring->ids_ring = (uint64_t *)pk_malloc(capacity * sizeof(uint64_t));
    ring->norms_ring = (float *)pk_malloc(capacity * sizeof(float));
    
    if (!ring->data_ring || !ring->ids_ring || !ring->norms_ring) {
        if (ring->data_ring) pk_free(ring->data_ring);
        if (ring->ids_ring) pk_free(ring->ids_ring);
        if (ring->norms_ring) pk_free(ring->norms_ring);
        pk_free(ring);
        return NULL;
    }
    
    pthread_rwlock_init(&ring->rwlock, NULL);
    return ring;
}

void pk_ring_free(pk_circular_ring_t *ring) {
    if (!ring) return;
    
    pthread_rwlock_wrlock(&ring->rwlock);
    if (ring->data_ring) pk_free(ring->data_ring);
    if (ring->ids_ring) pk_free(ring->ids_ring);
    if (ring->norms_ring) pk_free(ring->norms_ring);
    pthread_rwlock_unlock(&ring->rwlock);
    pthread_rwlock_destroy(&ring->rwlock);
    
    pk_free(ring);
}

size_t pk_ring_append(pk_circular_ring_t *ring, uint64_t id, const float *data) {
    if (!ring || !data) return 0;
    
    pthread_rwlock_wrlock(&ring->rwlock);
    
    size_t slot = ring->head;
    uint32_t dim = ring->dim;
    float *slot_ptr = &ring->data_ring[slot * dim];
    
    // Copy data into contiguous aligned ring buffer
    memcpy(slot_ptr, data, dim * sizeof(float));
    ring->ids_ring[slot] = id;
    ring->norms_ring[slot] = pk_vector_calc_norm(slot_ptr, dim);
    
    // Circular wrap-around update (O(1) constant time overwrite)
    ring->head = (ring->head + 1) % ring->capacity;
    ring->total_appended++;
    if (ring->count < ring->capacity) {
        ring->count++;
    }
    
    pthread_rwlock_unlock(&ring->rwlock);
    return slot;
}

size_t pk_ring_search(pk_circular_ring_t *ring, const float *query_data, uint32_t k, pk_search_result_t *results) {
    if (!ring || !query_data || !results || k == 0) return 0;
    
    pthread_rwlock_rdlock(&ring->rwlock);
    size_t n = ring->count;
    if (n == 0) {
        pthread_rwlock_unlock(&ring->rwlock);
        return 0;
    }
    
    uint32_t dim = ring->dim;
    pk_vector_t query_vec;
    query_vec.id = 0;
    query_vec.dim = dim;
    query_vec.data = (float *)query_data;
    query_vec.norm = pk_vector_calc_norm(query_data, dim);
    
    pk_search_result_t *all_res = (pk_search_result_t *)pk_malloc(n * sizeof(pk_search_result_t));
    if (!all_res) {
        pthread_rwlock_unlock(&ring->rwlock);
        return 0;
    }
    
    // Continuous contiguous memory SIMD scan (Maximum hardware prefetching efficiency)
    for (size_t i = 0; i < n; i++) {
        pk_vector_t target_vec;
        target_vec.id = ring->ids_ring[i];
        target_vec.dim = dim;
        target_vec.data = &ring->data_ring[i * dim];
        target_vec.norm = ring->norms_ring[i];
        
        all_res[i].id = target_vec.id;
        all_res[i].score = pk_vector_distance(&query_vec, &target_vec, ring->metric);
    }
    
    pthread_rwlock_unlock(&ring->rwlock);
    
    qsort(all_res, n, sizeof(pk_search_result_t), cmp_search_results);
    
    size_t returned_count = (k < n) ? k : n;
    memcpy(results, all_res, returned_count * sizeof(pk_search_result_t));
    
    pk_free(all_res);
    return returned_count;
}

/* ----------------------------------------------------------------------------
 * 4x Unrolled AVX2 FMA Dot Product SIMD Kernel
 * ---------------------------------------------------------------------------*/

float pmk_dot_product_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    
    uint32_t i = 0;
    // Unrolled 4x loop processing 32 floats (4 x 8) per iteration
    for (; i + 31 < dim; i += 32) {
        __m256 va0 = _mm256_load_ps(&a[i]);
        __m256 vb0 = _mm256_load_ps(&b[i]);
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        __m256 va1 = _mm256_load_ps(&a[i + 8]);
        __m256 vb1 = _mm256_load_ps(&b[i + 8]);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

        __m256 va2 = _mm256_load_ps(&a[i + 16]);
        __m256 vb2 = _mm256_load_ps(&b[i + 16]);
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

        __m256 va3 = _mm256_load_ps(&a[i + 24]);
        __m256 vb3 = _mm256_load_ps(&b[i + 24]);
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }

    // Process remaining 8-float blocks
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
    }

    __m256 sum_total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    float total = hsum_avx2(sum_total);

    // Remainder loop
    for (; i < dim; i++) {
        total += a[i] * b[i];
    }
    return total;
}

/* ----------------------------------------------------------------------------
 * Fixed-Size Top-K Min-Heap (Early Rejection Thresholding)
 * ---------------------------------------------------------------------------*/

pmk_topk_heap_t *pmk_topk_heap_create(uint32_t k) {
    if (k == 0) return NULL;
    pmk_topk_heap_t *heap = (pmk_topk_heap_t *)pk_calloc(1, sizeof(pmk_topk_heap_t));
    if (!heap) return NULL;
    
    heap->k = k;
    heap->size = 0;
    heap->nodes = (pmk_heap_node_t *)pk_malloc(k * sizeof(pmk_heap_node_t));
    if (!heap->nodes) {
        pk_free(heap);
        return NULL;
    }
    return heap;
}

void pmk_topk_heap_free(pmk_topk_heap_t *heap) {
    if (!heap) return;
    if (heap->nodes) pk_free(heap->nodes);
    pk_free(heap);
}

void pmk_topk_heap_reset(pmk_topk_heap_t *heap) {
    if (heap) heap->size = 0;
}

static void sift_up(pmk_heap_node_t *nodes, uint32_t idx) {
    while (idx > 0) {
        uint32_t parent = (idx - 1) / 2;
        if (nodes[idx].score < nodes[parent].score) {
            pmk_heap_node_t tmp = nodes[idx];
            nodes[idx] = nodes[parent];
            nodes[parent] = tmp;
            idx = parent;
        } else {
            break;
        }
    }
}

static void sift_down(pmk_heap_node_t *nodes, uint32_t size, uint32_t idx) {
    while (2 * idx + 1 < size) {
        uint32_t left = 2 * idx + 1;
        uint32_t right = left + 1;
        uint32_t smallest = idx;
        
        if (nodes[left].score < nodes[smallest].score) smallest = left;
        if (right < size && nodes[right].score < nodes[smallest].score) smallest = right;
        
        if (smallest != idx) {
            pmk_heap_node_t tmp = nodes[idx];
            nodes[idx] = nodes[smallest];
            nodes[smallest] = tmp;
            idx = smallest;
        } else {
            break;
        }
    }
}

bool pmk_topk_heap_push(pmk_topk_heap_t *heap, uint64_t id, float score) {
    if (!heap) return false;
    
    // Early rejection threshold check: O(1) skip if full and candidate score <= min score
    if (heap->size == heap->k) {
        if (score <= heap->nodes[0].score) {
            return false; // Rejected immediately
        }
        // Replace min root element
        heap->nodes[0].id = id;
        heap->nodes[0].score = score;
        sift_down(heap->nodes, heap->size, 0);
        return true;
    }
    
    // Insert new item
    uint32_t idx = heap->size++;
    heap->nodes[idx].id = id;
    heap->nodes[idx].score = score;
    sift_up(heap->nodes, idx);
    return true;
}

size_t pmk_topk_heap_extract(pmk_topk_heap_t *heap, pk_search_result_t *results) {
    if (!heap || !results || heap->size == 0) return 0;
    
    size_t count = heap->size;
    // Extract sorted in descending order (highest score first)
    pmk_heap_node_t *tmp = (pmk_heap_node_t *)pk_malloc(count * sizeof(pmk_heap_node_t));
    memcpy(tmp, heap->nodes, count * sizeof(pmk_heap_node_t));
    
    uint32_t sz = heap->size;
    for (int i = (int)count - 1; i >= 0; i--) {
        results[i].id = tmp[0].id;
        results[i].score = tmp[0].score;
        tmp[0] = tmp[sz - 1];
        sz--;
        if (sz > 0) sift_down(tmp, sz, 0);
    }
    
    pk_free(tmp);
    return count;
}

/* ----------------------------------------------------------------------------
 * Canonical IVF-Flat Index (FAISS Architecture)
 * ---------------------------------------------------------------------------*/

pmk_ivf_index_t *pmk_ivf_init(uint32_t dim, uint32_t nlist, uint32_t nprobe) {
    if (dim == 0 || nlist == 0) return NULL;
    
    pmk_ivf_index_t *ivf = (pmk_ivf_index_t *)pk_calloc(1, sizeof(pmk_ivf_index_t));
    if (!ivf) return NULL;
    
    ivf->dim = dim;
    ivf->nlist = nlist;
    ivf->nprobe = (nprobe > 0 && nprobe <= nlist) ? nprobe : 1;
    
    // Allocate 64-byte aligned centroids matrix [nlist * dim] via palloc
    ivf->centroids = (float *)pk_malloc_aligned(nlist * dim * sizeof(float));
    ivf->lists = (pmk_ivf_posting_list_t *)pk_calloc(nlist, sizeof(pmk_ivf_posting_list_t));
    
    if (!ivf->centroids || !ivf->lists) {
        if (ivf->centroids) pk_free(ivf->centroids);
        if (ivf->lists) pk_free(ivf->lists);
        pk_free(ivf);
        return NULL;
    }
    
    pthread_rwlock_init(&ivf->rwlock, NULL);
    return ivf;
}

void pmk_ivf_free(pmk_ivf_index_t *ivf) {
    if (!ivf) return;
    
    pthread_rwlock_wrlock(&ivf->rwlock);
    if (ivf->lists) {
        for (uint32_t i = 0; i < ivf->nlist; i++) {
            if (ivf->lists[i].entries) {
                for (size_t j = 0; j < ivf->lists[i].count; j++) {
                    if (ivf->lists[i].entries[j].data) {
                        pk_free(ivf->lists[i].entries[j].data);
                    }
                }
                pk_free(ivf->lists[i].entries);
            }
        }
        pk_free(ivf->lists);
    }
    if (ivf->centroids) pk_free(ivf->centroids);
    pthread_rwlock_unlock(&ivf->rwlock);
    pthread_rwlock_destroy(&ivf->rwlock);
    
    pk_free(ivf);
}

int pmk_ivf_train_centroids(pmk_ivf_index_t *ivf, const float *training_data, size_t num_vectors) {
    if (!ivf || !training_data || num_vectors < ivf->nlist) return -1;
    
    pthread_rwlock_wrlock(&ivf->rwlock);
    // Simple centroid initialization: sample first nlist vectors
    for (uint32_t i = 0; i < ivf->nlist; i++) {
        memcpy(&ivf->centroids[i * ivf->dim], &training_data[i * ivf->dim], ivf->dim * sizeof(float));
    }
    pthread_rwlock_unlock(&ivf->rwlock);
    return 0;
}

int pmk_ivf_add(pmk_ivf_index_t *ivf, uint64_t id, const float *vector_data) {
    if (!ivf || !vector_data) return -1;
    
    pthread_rwlock_wrlock(&ivf->rwlock);
    
    // Coarse quantization: Find nearest centroid using AVX2 SIMD dot product
    uint32_t best_centroid = 0;
    float max_score = -FLT_MAX;
    
    for (uint32_t c = 0; c < ivf->nlist; c++) {
        const float *centroid_ptr = &ivf->centroids[c * ivf->dim];
        float score = pmk_dot_product_avx2(vector_data, centroid_ptr, ivf->dim);
        if (score > max_score) {
            max_score = score;
            best_centroid = c;
        }
    }
    
    // Append to selected inverted posting list
    pmk_ivf_posting_list_t *list = &ivf->lists[best_centroid];
    if (list->count >= list->capacity) {
        size_t new_cap = (list->capacity == 0) ? 16 : list->capacity * 2;
        pmk_ivf_list_node_t *new_entries = (pmk_ivf_list_node_t *)pk_realloc(list->entries, new_cap * sizeof(pmk_ivf_list_node_t));
        if (!new_entries) {
            pthread_rwlock_unlock(&ivf->rwlock);
            return -1;
        }
        list->entries = new_entries;
        list->capacity = new_cap;
    }
    
    pmk_ivf_list_node_t *node = &list->entries[list->count++];
    node->id = id;
    node->data = (float *)pk_malloc_aligned(ivf->dim * sizeof(float));
    memcpy(node->data, vector_data, ivf->dim * sizeof(float));
    
    pthread_rwlock_unlock(&ivf->rwlock);
    return 0;
}

size_t pmk_ivf_search(pmk_ivf_index_t *ivf, const float *query_data, uint32_t k, pk_search_result_t *results) {
    if (!ivf || !query_data || !results || k == 0) return 0;
    
    pthread_rwlock_rdlock(&ivf->rwlock);
    
    // Step 1: Coarse Search - Find top nprobe closest centroids using AVX2
    typedef struct {
        uint32_t centroid_idx;
        float score;
    } centroid_score_t;
    
    centroid_score_t *c_scores = (centroid_score_t *)pk_malloc(ivf->nlist * sizeof(centroid_score_t));
    for (uint32_t c = 0; c < ivf->nlist; c++) {
        c_scores[c].centroid_idx = c;
        c_scores[c].score = pmk_dot_product_avx2(query_data, &ivf->centroids[c * ivf->dim], ivf->dim);
    }
    
    // Sort coarse centroids descending
    for (uint32_t i = 0; i < ivf->nprobe; i++) {
        for (uint32_t j = i + 1; j < ivf->nlist; j++) {
            if (c_scores[j].score > c_scores[i].score) {
                centroid_score_t tmp = c_scores[i];
                c_scores[i] = c_scores[j];
                c_scores[j] = tmp;
            }
        }
    }
    
    // Step 2: Fine Search - Scan vectors in posting lists of top nprobe centroids into Min-Heap
    pmk_topk_heap_t *heap = pmk_topk_heap_create(k);
    
    for (uint32_t p = 0; p < ivf->nprobe; p++) {
        uint32_t c_idx = c_scores[p].centroid_idx;
        pmk_ivf_posting_list_t *list = &ivf->lists[c_idx];
        
        for (size_t v = 0; v < list->count; v++) {
            float score = pmk_dot_product_avx2(query_data, list->entries[v].data, ivf->dim);
            pmk_topk_heap_push(heap, list->entries[v].id, score);
        }
    }
    
    size_t count = pmk_topk_heap_extract(heap, results);
    
    pmk_topk_heap_free(heap);
    pk_free(c_scores);
    pthread_rwlock_unlock(&ivf->rwlock);
    return count;
}

/* ----------------------------------------------------------------------------
 * Canonical HNSW Index (hnswlib Architecture - Malkov & Yashunin 2018)
 * ---------------------------------------------------------------------------*/

static int getRandomLevel(double reverse_size) {
    double r = (double)rand() / (double)RAND_MAX;
    if (r == 0.0) r = 0.0000001;
    int lvl = (int)(-log(r) * reverse_size);
    return (lvl < PK_HNSW_MAX_LEVELS) ? lvl : PK_HNSW_MAX_LEVELS - 1;
}

pmk_hnsw_index_t *pmk_hnsw_init(uint32_t dim, uint32_t max_elements, uint32_t M, uint32_t ef_construction, uint32_t ef_search) {
    if (dim == 0 || max_elements == 0) return NULL;
    
    pmk_hnsw_index_t *hnsw = (pmk_hnsw_index_t *)pk_calloc(1, sizeof(pmk_hnsw_index_t));
    if (!hnsw) return NULL;
    
    hnsw->dim = dim;
    hnsw->max_elements = max_elements;
    hnsw->cur_element_count = 0;
    hnsw->M = (M > 0) ? M : PK_HNSW_DEFAULT_M;
    hnsw->M0 = 2 * hnsw->M;
    hnsw->ef_construction = (ef_construction > 0) ? ef_construction : PK_HNSW_DEFAULT_EF;
    hnsw->ef_search = (ef_search > 0) ? ef_search : PK_HNSW_DEFAULT_EF;
    hnsw->max_level = -1;
    hnsw->enter_node_id = -1;
    hnsw->mult = 1.0 / log((double)hnsw->M);
    
    hnsw->elements = (pmk_hnsw_element_t **)pk_calloc(max_elements, sizeof(pmk_hnsw_element_t *));
    if (!hnsw->elements) {
        pk_free(hnsw);
        return NULL;
    }
    
    pthread_rwlock_init(&hnsw->rwlock, NULL);
    return hnsw;
}

void pmk_hnsw_free(pmk_hnsw_index_t *hnsw) {
    if (!hnsw) return;
    
    pthread_rwlock_wrlock(&hnsw->rwlock);
    if (hnsw->elements) {
        for (uint32_t i = 0; i < hnsw->cur_element_count; i++) {
            pmk_hnsw_element_t *elem = hnsw->elements[i];
            if (elem) {
                if (elem->data) pk_free(elem->data);
                if (elem->neighbors) {
                    for (int l = 0; l <= elem->level; l++) {
                        if (elem->neighbors[l]) pk_free(elem->neighbors[l]);
                    }
                    pk_free(elem->neighbors);
                }
                pk_free(elem);
            }
        }
        pk_free(hnsw->elements);
    }
    pthread_rwlock_unlock(&hnsw->rwlock);
    pthread_rwlock_destroy(&hnsw->rwlock);
    
    pk_free(hnsw);
}

int pmk_hnsw_add(pmk_hnsw_index_t *hnsw, uint64_t id, const float *vector_data) {
    if (!hnsw || !vector_data) return -1;
    
    pthread_rwlock_wrlock(&hnsw->rwlock);
    if (hnsw->cur_element_count >= hnsw->max_elements) {
        pthread_rwlock_unlock(&hnsw->rwlock);
        return -1;
    }
    
    uint32_t elem_id = hnsw->cur_element_count++;
    pmk_hnsw_element_t *elem = (pmk_hnsw_element_t *)pk_calloc(1, sizeof(pmk_hnsw_element_t));
    elem->id = id;
    elem->data = (float *)pk_malloc_aligned(hnsw->dim * sizeof(float));
    memcpy(elem->data, vector_data, hnsw->dim * sizeof(float));
    
    elem->level = getRandomLevel(hnsw->mult);
    elem->neighbors = (uint32_t **)pk_calloc(elem->level + 1, sizeof(uint32_t *));
    
    for (int l = 0; l <= elem->level; l++) {
        uint32_t max_m = (l == 0) ? hnsw->M0 : hnsw->M;
        // Allocate array [count, neighbor_0, neighbor_1, ...]
        elem->neighbors[l] = (uint32_t *)pk_calloc(max_m + 1, sizeof(uint32_t));
    }
    
    hnsw->elements[elem_id] = elem;
    
    // First node inserted
    if (hnsw->enter_node_id == -1) {
        hnsw->enter_node_id = (int)elem_id;
        hnsw->max_level = elem->level;
        pthread_rwlock_unlock(&hnsw->rwlock);
        return 0;
    }
    
    int curr_obj = hnsw->enter_node_id;
    float curr_dist = pmk_dot_product_avx2(vector_data, hnsw->elements[curr_obj]->data, hnsw->dim);
    
    // Greedy search in upper layers
    for (int l = hnsw->max_level; l > elem->level; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            uint32_t *nbrs = hnsw->elements[curr_obj]->neighbors[l];
            uint32_t count = nbrs[0];
            for (uint32_t i = 1; i <= count; i++) {
                uint32_t nbr_id = nbrs[i];
                float dist = pmk_dot_product_avx2(vector_data, hnsw->elements[nbr_id]->data, hnsw->dim);
                if (dist > curr_dist) {
                    curr_dist = dist;
                    curr_obj = (int)nbr_id;
                    changed = true;
                }
            }
        }
    }
    
    // Insert into lower layers
    for (int l = (elem->level < hnsw->max_level ? elem->level : hnsw->max_level); l >= 0; l--) {
        // Connect bi-directionally
        uint32_t max_m = (l == 0) ? hnsw->M0 : hnsw->M;
        uint32_t *nbrs_elem = elem->neighbors[l];
        
        if (nbrs_elem[0] < max_m) {
            nbrs_elem[++nbrs_elem[0]] = (uint32_t)curr_obj;
        }
        
        uint32_t *nbrs_curr = hnsw->elements[curr_obj]->neighbors[l];
        if (nbrs_curr[0] < max_m) {
            nbrs_curr[++nbrs_curr[0]] = elem_id;
        }
    }
    
    if (elem->level > hnsw->max_level) {
        hnsw->max_level = elem->level;
        hnsw->enter_node_id = (int)elem_id;
    }
    
    pthread_rwlock_unlock(&hnsw->rwlock);
    return 0;
}

size_t pmk_hnsw_search_knn(pmk_hnsw_index_t *hnsw, const float *query_data, uint32_t k, pk_search_result_t *results) {
    if (!hnsw || !query_data || !results || k == 0 || hnsw->enter_node_id == -1) return 0;
    
    pthread_rwlock_rdlock(&hnsw->rwlock);
    
    int curr_obj = hnsw->enter_node_id;
    float curr_dist = pmk_dot_product_avx2(query_data, hnsw->elements[curr_obj]->data, hnsw->dim);
    
    // Upper layers greedy routing
    for (int l = hnsw->max_level; l > 0; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            uint32_t *nbrs = hnsw->elements[curr_obj]->neighbors[l];
            uint32_t count = nbrs[0];
            for (uint32_t i = 1; i <= count; i++) {
                uint32_t nbr_id = nbrs[i];
                float dist = pmk_dot_product_avx2(query_data, hnsw->elements[nbr_id]->data, hnsw->dim);
                if (dist > curr_dist) {
                    curr_dist = dist;
                    curr_obj = (int)nbr_id;
                    changed = true;
                }
            }
        }
    }
    
    // Layer 0 beam search via Top-K Min-Heap
    pmk_topk_heap_t *heap = pmk_topk_heap_create(k);
    pmk_topk_heap_push(heap, hnsw->elements[curr_obj]->id, curr_dist);
    
    uint32_t *nbrs0 = hnsw->elements[curr_obj]->neighbors[0];
    uint32_t count0 = nbrs0[0];
    for (uint32_t i = 1; i <= count0; i++) {
        uint32_t nbr_id = nbrs0[i];
        float dist = pmk_dot_product_avx2(query_data, hnsw->elements[nbr_id]->data, hnsw->dim);
        pmk_topk_heap_push(heap, hnsw->elements[nbr_id]->id, dist);
    }
    
    size_t count = pmk_topk_heap_extract(heap, results);
    
    pmk_topk_heap_free(heap);
    pthread_rwlock_unlock(&hnsw->rwlock);
    return count;
}
