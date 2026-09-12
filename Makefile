CC ?= gcc
CFLAGS ?= -O3 -march=native -mavx2 -mfma -Wall -Wextra -D_GNU_SOURCE -I./palloc/include -I./liburing/src/include
LDFLAGS ?= ./palloc/build/libpalloc.a ./liburing/src/liburing.a -lm -lpthread -lrt -latomic

SRCS = src/pomaikache.c src/io_engine.c src/main.c
OBJS = $(SRCS:.c=.o)
TARGET = bin/pomaikache

.PHONY: all submodules clean bench test

all: submodules $(TARGET)

submodules:
	@if [ ! -f liburing/src/liburing.a ]; then \
		echo "Building liburing..."; \
		$(MAKE) -C liburing -j$(shell nproc); \
	fi
	@if [ ! -f palloc/build/libpalloc.a ]; then \
		echo "Building palloc..."; \
		cmake -B palloc/build -S palloc -DPA_BUILD_TESTS=OFF -DPA_BUILD_SHARED=OFF; \
		cmake --build palloc/build -j$(shell nproc); \
	fi

$(TARGET): $(OBJS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "[pomaikache] Built executable successfully: $(TARGET)"

TEST_SRCS = tests/test_math.c tests/test_ring_eviction.c tests/test_memory_alignment.c tests/test_concurrency.c tests/test_network_chaos.c tests/test_performance.c
TEST_BINS = bin/test_math bin/test_ring_eviction bin/test_memory_alignment bin/test_concurrency bin/test_network_chaos bin/test_performance

bin/test_math: src/pomaikache.o tests/test_math.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_ring_eviction: src/pomaikache.o tests/test_ring_eviction.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_memory_alignment: src/pomaikache.o tests/test_memory_alignment.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_concurrency: src/pomaikache.o tests/test_concurrency.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_network_chaos: src/pomaikache.o tests/test_network_chaos.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_performance: src/pomaikache.o tests/test_performance.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_suite: src/pomaikache.o src/test_suite.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "[pomaikache] Built test suite: bin/test_suite"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJS) src/test_suite.o tests/*.o bin/pomaikache bin/test_*

bench: $(TARGET)
	./$(TARGET) --bench

test: $(TEST_BINS)
	./bin/test_math
	./bin/test_ring_eviction
	./bin/test_memory_alignment
	./bin/test_concurrency
	./bin/test_network_chaos
	./bin/test_performance
