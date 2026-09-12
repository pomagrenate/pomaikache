# Multi-stage Dockerfile for Pomaikache Vector Cache
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and compiler dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project source and submodules
COPY . .

# Build submodules (palloc & liburing) and pomaikache binary
RUN make submodules && make clean && make -j$(nproc)

# Lightweight runtime container
FROM ubuntu:22.04 AS runner

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/bin/pomaikache /app/pomaikache

# Expose default pomaikache port
EXPOSE 9090

# Persistence volume for WAL and snapshots
VOLUME ["/app/data"]

ENTRYPOINT ["/app/pomaikache"]
CMD ["--port", "9090", "--dim", "128", "--capacity", "10000"]
