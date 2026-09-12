/* ----------------------------------------------------------------------------
 * Pomaikache - High-Performance Vector Caching & Retrieval Engine in C
 * liburing Network Server & Asynchronous Persistence Engine
 * ---------------------------------------------------------------------------*/

#include "pomaikache.h"

enum io_type {
    IO_ACCEPT = 0,
    IO_READ   = 1,
    IO_WRITE  = 2
};

typedef struct {
    enum io_type type;
    int fd;
    char buf[PK_MAX_BUF_SIZE];
    size_t buf_len;
} pk_io_request_t;

/* ----------------------------------------------------------------------------
 * Engine Creation & Initialization
 * ---------------------------------------------------------------------------*/

pk_engine_t *pk_engine_create(const pk_config_t *config) {
    pk_engine_t *engine = (pk_engine_t *)pk_calloc(1, sizeof(pk_engine_t));
    if (!engine) return NULL;
    
    if (config) {
        memcpy(&engine->config, config, sizeof(pk_config_t));
    } else {
        engine->config.port = 9090;
        engine->config.vector_dim = 128;
        engine->config.lru_capacity = 10000;
        engine->config.metric = PK_METRIC_COSINE;
        strncpy(engine->config.wal_path, "pomaikache.wal", sizeof(engine->config.wal_path));
        strncpy(engine->config.snapshot_path, "pomaikache.snap", sizeof(engine->config.snapshot_path));
    }
    
    // Allocate thread-local palloc heap for the engine
    engine->heap = pk_heap_new();
    
    // Create L1 Cache & L2 Index
    engine->cache = pk_lru_create(engine->config.lru_capacity);
    engine->index = pk_index_create(engine->config.vector_dim, engine->config.metric, false);
    engine->running = false;
    
    return engine;
}

void pk_engine_free(pk_engine_t *engine) {
    if (!engine) return;
    
    if (engine->cache) pk_lru_free(engine->cache);
    if (engine->index) pk_index_free(engine->index);
    if (engine->wal_fd > 0) close(engine->wal_fd);
    if (engine->server_fd > 0) close(engine->server_fd);
    
    io_uring_queue_exit(&engine->ring);
    if (engine->heap) pk_heap_delete(engine->heap);
    
    pk_free(engine);
}

/* ----------------------------------------------------------------------------
 * WAL Journaling & Persistence (liburing)
 * ---------------------------------------------------------------------------*/

int pk_wal_init(pk_engine_t *engine) {
    if (!engine) return -1;
    
    engine->wal_fd = open(engine->config.wal_path, O_CREAT | O_RDWR | O_APPEND, 0644);
    if (engine->wal_fd < 0) {
        perror("Failed to open WAL file");
        return -1;
    }
    return 0;
}

int pk_wal_append(pk_engine_t *engine, const pk_vector_t *vec) {
    if (!engine || !vec || engine->wal_fd < 0) return -1;
    
    // Binary layout: [uint64_t id][uint32_t dim][float data[dim]]
    size_t data_size = vec->dim * sizeof(float);
    size_t total_size = sizeof(uint64_t) + sizeof(uint32_t) + data_size;
    
    uint8_t *buf = (uint8_t *)pk_malloc(total_size);
    if (!buf) return -1;
    
    uint8_t *p = buf;
    memcpy(p, &vec->id, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, &vec->dim, sizeof(uint32_t)); p += sizeof(uint32_t);
    memcpy(p, vec->data, data_size);
    
    ssize_t written = write(engine->wal_fd, buf, total_size);
    pk_free(buf);
    
    return (written == (ssize_t)total_size) ? 0 : -1;
}

int pk_wal_recover(pk_engine_t *engine) {
    if (!engine || engine->wal_fd < 0) return -1;
    
    lseek(engine->wal_fd, 0, SEEK_SET);
    
    uint64_t id;
    uint32_t dim;
    size_t count = 0;
    
    while (read(engine->wal_fd, &id, sizeof(uint64_t)) == sizeof(uint64_t)) {
        if (read(engine->wal_fd, &dim, sizeof(uint32_t)) != sizeof(uint32_t)) break;
        
        float *data = (float *)pk_malloc(dim * sizeof(float));
        if (read(engine->wal_fd, data, dim * sizeof(float)) != (ssize_t)(dim * sizeof(float))) {
            pk_free(data);
            break;
        }
        
        pk_vector_t *vec = pk_vector_create(id, dim, data);
        pk_free(data);
        
        if (vec) {
            pk_index_insert(engine->index, vec);
            count++;
        }
    }
    
    printf("[pomaikache] WAL recovered %zu vectors\n", count);
    return 0;
}

/* ----------------------------------------------------------------------------
 * Snapshot Save & Load
 * ---------------------------------------------------------------------------*/

int pk_snapshot_save(pk_engine_t *engine, const char *filepath) {
    if (!engine || !filepath) return -1;
    
    int fd = open(filepath, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) return -1;
    
    pthread_rwlock_rdlock(&engine->index->rwlock);
    uint32_t dim = engine->index->dim;
    size_t count = engine->index->count;
    
    if (write(fd, &dim, sizeof(uint32_t)) < 0 ||
        write(fd, &count, sizeof(size_t)) < 0) {
        pthread_rwlock_unlock(&engine->index->rwlock);
        close(fd);
        return -1;
    }
    
    for (size_t i = 0; i < count; i++) {
        pk_vector_t *vec = engine->index->vectors[i];
        if (write(fd, &vec->id, sizeof(uint64_t)) < 0 ||
            write(fd, vec->data, dim * sizeof(float)) < 0) {
            break;
        }
    }
    
    pthread_rwlock_unlock(&engine->index->rwlock);
    close(fd);
    
    printf("[pomaikache] Snapshot saved (%zu vectors to %s)\n", count, filepath);
    return 0;
}

int pk_snapshot_load(pk_engine_t *engine, const char *filepath) {
    if (!engine || !filepath) return -1;
    
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return -1;
    
    uint32_t dim;
    size_t count;
    
    if (read(fd, &dim, sizeof(uint32_t)) != sizeof(uint32_t) ||
        read(fd, &count, sizeof(size_t)) != sizeof(size_t)) {
        close(fd);
        return -1;
    }
    
    float *data_buf = (float *)pk_malloc(dim * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        uint64_t id;
        if (read(fd, &id, sizeof(uint64_t)) != sizeof(uint64_t) ||
            read(fd, data_buf, dim * sizeof(float)) != (ssize_t)(dim * sizeof(float))) {
            break;
        }
        pk_vector_t *vec = pk_vector_create(id, dim, data_buf);
        if (vec) {
            pk_index_insert(engine->index, vec);
        }
    }
    
    pk_free(data_buf);
    close(fd);
    
    printf("[pomaikache] Snapshot loaded (%zu vectors from %s)\n", count, filepath);
    return 0;
}

/* ----------------------------------------------------------------------------
 * liburing Network Server Engine
 * ---------------------------------------------------------------------------*/

static void add_accept_req(struct io_uring *ring, int server_fd) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe) return;
    
    pk_io_request_t *req = (pk_io_request_t *)pk_malloc(sizeof(pk_io_request_t));
    req->type = IO_ACCEPT;
    req->fd = server_fd;
    
    io_uring_prep_accept(sqe, server_fd, NULL, NULL, 0);
    io_uring_sqe_set_data(sqe, req);
}

static void add_read_req(struct io_uring *ring, int client_fd) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe) return;
    
    pk_io_request_t *req = (pk_io_request_t *)pk_malloc(sizeof(pk_io_request_t));
    req->type = IO_READ;
    req->fd = client_fd;
    req->buf_len = 0;
    
    io_uring_prep_recv(sqe, client_fd, req->buf, PK_MAX_BUF_SIZE, 0);
    io_uring_sqe_set_data(sqe, req);
}

static void add_write_req(struct io_uring *ring, int client_fd, const char *msg, size_t len) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe) return;
    
    pk_io_request_t *req = (pk_io_request_t *)pk_malloc(sizeof(pk_io_request_t));
    req->type = IO_WRITE;
    req->fd = client_fd;
    req->buf_len = (len < PK_MAX_BUF_SIZE) ? len : PK_MAX_BUF_SIZE;
    memcpy(req->buf, msg, req->buf_len);
    
    io_uring_prep_send(sqe, client_fd, req->buf, req->buf_len, 0);
    io_uring_sqe_set_data(sqe, req);
}

int pk_engine_start_server(pk_engine_t *engine) {
    if (!engine) return -1;
    
    // Initialize io_uring
    if (io_uring_queue_init(PK_QUEUE_DEPTH, &engine->ring, 0) < 0) {
        perror("io_uring_queue_init failed");
        return -1;
    }
    
    // Create server TCP socket
    engine->server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (engine->server_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    int opt = 1;
    setsockopt(engine->server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(engine->config.port);
    
    if (bind(engine->server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind failed");
        return -1;
    }
    
    if (listen(engine->server_fd, 128) < 0) {
        perror("Listen failed");
        return -1;
    }
    
    printf("[pomaikache] Server started on port %d using liburing\n", engine->config.port);
    engine->running = true;
    
    add_accept_req(&engine->ring, engine->server_fd);
    
    while (engine->running) {
        io_uring_submit(&engine->ring);
        
        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&engine->ring, &cqe);
        if (ret < 0) {
            if (ret == -EINTR) continue;
            break;
        }
        
        pk_io_request_t *req = (pk_io_request_t *)io_uring_cqe_get_data(cqe);
        if (!req) {
            io_uring_cqe_seen(&engine->ring, cqe);
            continue;
        }
        
        if (cqe->res < 0) {
            if (req->type == IO_READ || req->type == IO_WRITE) {
                close(req->fd);
            }
            pk_free(req);
            io_uring_cqe_seen(&engine->ring, cqe);
            continue;
        }
        
        switch (req->type) {
            case IO_ACCEPT: {
                int client_fd = cqe->res;
                add_read_req(&engine->ring, client_fd);
                add_accept_req(&engine->ring, engine->server_fd);
                break;
            }
            case IO_READ: {
                int client_fd = req->fd;
                int bytes_read = cqe->res;
                
                if (bytes_read <= 0) {
                    close(client_fd);
                } else {
                    req->buf[bytes_read] = '\0';
                    char *cmd = req->buf;
                    // Strip trailing newlines
                    size_t len = strlen(cmd);
                    while (len > 0 && (cmd[len-1] == '\r' || cmd[len-1] == '\n')) {
                        cmd[--len] = '\0';
                    }
                    
                    if (strncmp(cmd, "PING", 4) == 0) {
                        add_write_req(&engine->ring, client_fd, "+PONG\r\n", 7);
                    } else if (strncmp(cmd, "STATS", 5) == 0) {
                        char resp[256];
                        snprintf(resp, sizeof(resp), "+OK vectors=%zu cache_hits=%lu cache_misses=%lu expired=%lu\r\n",
                                 engine->index->count, engine->cache->hits, engine->cache->misses, engine->cache->expired_count);
                        add_write_req(&engine->ring, client_fd, resp, strlen(resp));
                    } else if (strncmp(cmd, "PURGE", 5) == 0) {
                        size_t purged = pk_lru_purge_expired(engine->cache);
                        char resp[128];
                        snprintf(resp, sizeof(resp), "+OK purged=%zu\r\n", purged);
                        add_write_req(&engine->ring, client_fd, resp, strlen(resp));
                    } else if (strncmp(cmd, "DEL ", 4) == 0) {
                        const char *key = cmd + 4;
                        bool ok = pk_lru_del(engine->cache, key);
                        if (ok) add_write_req(&engine->ring, client_fd, "+OK deleted\r\n", 13);
                        else add_write_req(&engine->ring, client_fd, "-ERR key_not_found\r\n", 20);
                    } else if (strncmp(cmd, "DELID ", 6) == 0) {
                        uint64_t id = (uint64_t)atoll(cmd + 6);
                        bool ok = pk_index_remove(engine->index, id);
                        if (ok) add_write_req(&engine->ring, client_fd, "+OK deleted\r\n", 13);
                        else add_write_req(&engine->ring, client_fd, "-ERR id_not_found\r\n", 19);
                    } else if (strncmp(cmd, "GET ", 4) == 0) {
                        const char *key = cmd + 4;
                        pk_vector_t *vec = pk_lru_get(engine->cache, key);
                        if (vec) {
                            char resp[128];
                            snprintf(resp, sizeof(resp), "+OK id=%lu dim=%u norm=%.4f\r\n", vec->id, vec->dim, vec->norm);
                            add_write_req(&engine->ring, client_fd, resp, strlen(resp));
                        } else {
                            add_write_req(&engine->ring, client_fd, "-ERR cache_miss\r\n", 17);
                        }
                    } else if (strncmp(cmd, "PUSH ", 5) == 0 || strncmp(cmd, "SET ", 4) == 0) {
                        // Protocol: PUSH <key> <id> <val1,val2,val3...>
                        char key[64];
                        uint64_t id = 0;
                        char vec_str[PK_MAX_BUF_SIZE];
                        int offset = (cmd[0] == 'P') ? 5 : 4;
                        
                        if (sscanf(cmd + offset, "%63s %lu %s", key, &id, vec_str) == 3) {
                            uint32_t dim = engine->config.vector_dim;
                            float *floats = (float *)pk_malloc(dim * sizeof(float));
                            
                            char *token = strtok(vec_str, ",");
                            uint32_t parsed_count = 0;
                            while (token && parsed_count < dim) {
                                floats[parsed_count++] = (float)atof(token);
                                token = strtok(NULL, ",");
                            }
                            
                            if (parsed_count == dim) {
                                pk_vector_t *vec = pk_vector_create(id, dim, floats);
                                pk_index_insert(engine->index, vec);
                                pk_lru_put(engine->cache, key, vec);
                                pk_wal_append(engine, vec);
                                
                                char resp[128];
                                snprintf(resp, sizeof(resp), "+OK inserted id=%lu\r\n", id);
                                add_write_req(&engine->ring, client_fd, resp, strlen(resp));
                            } else {
                                add_write_req(&engine->ring, client_fd, "-ERR dimension_mismatch\r\n", 25);
                            }
                            pk_free(floats);
                        } else {
                            add_write_req(&engine->ring, client_fd, "-ERR syntax_error usage: PUSH <key> <id> <v1,v2...>\r\n", 53);
                        }
                    } else if (strncmp(cmd, "SEARCH ", 7) == 0) {
                        // Protocol: SEARCH <k> <val1,val2,val3...>
                        uint32_t top_k = 5;
                        char vec_str[PK_MAX_BUF_SIZE];
                        if (sscanf(cmd + 7, "%u %s", &top_k, vec_str) == 2) {
                            uint32_t dim = engine->config.vector_dim;
                            float *floats = (float *)pk_malloc(dim * sizeof(float));
                            
                            char *token = strtok(vec_str, ",");
                            uint32_t parsed_count = 0;
                            while (token && parsed_count < dim) {
                                floats[parsed_count++] = (float)atof(token);
                                token = strtok(NULL, ",");
                            }
                            
                            if (parsed_count == dim) {
                                pk_vector_t *query = pk_vector_create(0, dim, floats);
                                pk_search_result_t *res = (pk_search_result_t *)pk_malloc(top_k * sizeof(pk_search_result_t));
                                size_t found = pk_index_search(engine->index, query, top_k, res);
                                
                                char resp[PK_MAX_BUF_SIZE];
                                int len = snprintf(resp, sizeof(resp), "+OK found=%zu results=[", found);
                                for (size_t r = 0; r < found; r++) {
                                    len += snprintf(resp + len, sizeof(resp) - len, "(id:%lu,score:%.4f)%s",
                                                     res[r].id, res[r].score, (r + 1 < found) ? "," : "");
                                }
                                snprintf(resp + len, sizeof(resp) - len, "]\r\n");
                                
                                add_write_req(&engine->ring, client_fd, resp, strlen(resp));
                                pk_vector_free(query);
                                pk_free(res);
                            } else {
                                add_write_req(&engine->ring, client_fd, "-ERR dimension_mismatch\r\n", 25);
                            }
                            pk_free(floats);
                        } else {
                            add_write_req(&engine->ring, client_fd, "-ERR syntax_error usage: SEARCH <k> <v1,v2...>\r\n", 49);
                        }
                    } else {
                        add_write_req(&engine->ring, client_fd, "-ERR unknown_command\r\n", 22);
                    }
                    add_read_req(&engine->ring, client_fd);
                }
                break;
            }
            case IO_WRITE: {
                // Done sending response
                break;
            }
        }
        
        pk_free(req);
        io_uring_cqe_seen(&engine->ring, cqe);
    }
    
    return 0;
}
