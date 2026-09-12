# Pomaikache Systems Benchmarking Specification & Methodology

This document outlines the enterprise benchmarking specification, hardware isolation controls, workload profiles, statistical metrics, and execution protocols used to evaluate **`pomaikache`** against industry-standard in-memory caching engines (**Redis 7.2 + RediSearch** and **DragonflyDB 1.14**).

---

## 1. Executive Benchmarking Matrix

| Evaluation Domain | Target Metric | Benchmark Method | Tools Used | Pass / Evaluation Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion Bandwidth** | Ops/Sec & MB/s write throughput | Continuous $100\%$ PUSH into $O(1)$ ring buffer & L1 LRU | `memtier_benchmark`, Custom C Harness | Pomaikache $\ge 3.0\times$ Redis ingestion throughput. |
| **Search Latency Profile** | p50, p90, p99, p99.9 latencies ($\mu\text{s}$) | Open-loop Poisson query dispatch ($K=1, 5, 10, 50$) | `hdrhistogram`, `memtier_benchmark` | Tail latency $p99 < 5.0\text{ ms}$ under $90\%$ saturation load. |
| **Search Recall@K** | Ground Truth Precision ($\%$) | Brute-force exact Flat Scan vs. IVF-Flat / HNSW | Python ground truth evaluator, `ANN-Benchmarks` | Recall@10 $\ge 99.0\%$ relative to exact scan. |
| **Memory Amplification** | RSS & bytes / million vectors | Memory footprint profiling across 10M vectors | `palloc` `pa_stats_print()`, `smaps` | Zero dynamic heap fragmentation ($0\%$ RSS bloat). |
| **Hardware Efficiency** | IPC & Cache Miss Rates | Hardware performance counter sampling during load | Linux `perf stat`, `perf record` | Low LLC miss rates & high Instructions Per Cycle (IPC). |

---

## 2. Hardware Controls & System Environment Isolation

To ensure statistical reproducibility and eliminate OS-level CPU throttling and NUMA latency jitter, the following hardware and kernel controls must be applied prior to executing benchmarks:

### A. CPU Governor & Turbo Boost Disabling
```bash
# Set all CPU cores to maximum frequency performance mode
sudo cpupower frequency-set -g performance

# Disable Intel Turbo Boost to prevent frequency scaling skew
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Disable AMD Core Performance Boost (if running on AMD EPYC/Ryzen)
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

### B. NUMA Node Binding & CPU Core Pinning
```bash
# Pin engine process strictly to NUMA Socket 0 with local memory node allocation
numactl --physcpubind=0-7 --membind=0 ./bin/pomaikache --port 9090 --dim 128 --capacity 100000

# Pin load generator client to separate CPU cores on Socket 0 (isolating client from engine)
taskset -c 8-15 memtier_benchmark -s 127.0.0.1 -p 9090 --protocol=redis
```

### C. Kernel Networking & `io_uring` Tuning
```bash
# Scale socket listen queue size for high TCP connection backlogs
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# Increase max socket read/write buffer allocations
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
```

---

## 3. Workload Profiles & Dataset Standards

### Vector Dimension Profiles
- **128 Dimensions** (SIFT-1M / Computer Vision embeddings)
- **384 Dimensions** (All-MiniLM-L6-v2 / Fast Sentence Transformers)
- **768 Dimensions** (BERT-base / OpenAI `text-embedding-3-small`)
- **1536 Dimensions** (OpenAI `ada-002` / Large LLM semantic embeddings)

### Access Distribution Patterns
1. **Uniform Random Distribution**: Baseline worst-case memory access pattern (tests raw SIMD memory bandwidth and cache miss tolerance).
2. **Zipfian / Power-Law Distribution ($\alpha = 0.99$)**: Simulates realistic production prompt caching where top $1\%$ of vectors receive $80\%$ of query volume.

### Workload Operation Mixes
- **Profile A (Pure Write / Ingestion)**: $100\%$ `PUSH` operations (evaluates ring buffer overwrite and `palloc` arena allocation speed).
- **Profile B (Pure Read / Retrieval)**: $100\%$ `SEARCH` Top-K ($K=10$) (evaluates AVX2 SIMD kernel compute and top-k heap sorting).
- **Profile C (Mixed Production Cache)**: $80\%$ `SEARCH` / $20\%$ `PUSH` with active TTL expirations ($50\text{ ms}$–$1000\text{ ms}$).

---

## 4. Hardware Telemetry Collection (`perf`)

During benchmark execution, CPU performance counters are logged using Linux `perf`:

```bash
# Monitor hardware performance counters over a 30-second benchmark run
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses \
  -p $(pgrep pomaikache) sleep 30
```

---

## 5. Step-by-Step Benchmark Execution Protocol

### Step 1: Cold System Isolation Setup
```bash
# 1. Clear OS page cache and swap
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# 2. Build pomaikache release target
make clean && make -j$(nproc)
```

### Step 2: Start Engine with Core Pinning
```bash
numactl --physcpubind=0-3 --membind=0 ./bin/pomaikache --port 9090 --dim 128 --capacity 100000
```

### Step 3: Execute Load Generator Sweep
```bash
# Run multi-threaded client load test across stepped concurrency levels (1, 8, 16, 64, 256 connections)
./bin/test_performance
```

---

## 6. Reproducibility & Open Source Artifact Verification

To reproduce all benchmark charts published in `README.md`:
```bash
# Generate high-resolution visual benchmark charts
python3 scripts/generate_charts.py
```
Outputs:
- `assets/chart_throughput.png`
- `assets/chart_latency.png`
- `assets/chart_recall.png`
