# Pomaikache QA Specification & Test Plan Matrix

## 1. Test Specification Matrix

| Test Case ID | Suite Category | Objective | Test Input | Invariant Condition | Pass/Fail Criteria |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-MATH-01** | Math Verification | Verify AVX2 FMA Dot Product precision vs. scalar reference | Random 128d, 384d, 768d, 1536d vectors | `\|dot_avx2 - dot_scalar\| <= 1e-5` | Pass if absolute delta $\le 10^{-5}$ across all dimensions. |
| **TC-MATH-02** | Math Verification | Cosine distance boundary verification | Orthogonal vectors ($A \cdot B = 0$), Identical vectors ($A \cdot B = 1$) | Cosine distance for identical = $0.0$, orthogonal = $1.0$ | Pass if distance equals mathematical expectation within float epsilon. |
| **TC-MATH-03** | Math Verification | Edge-case dimension & vector values | Zero vectors, Negative floats, NaN/Inf inputs | Zero vectors norm = $0.0$, NaN inputs safely rejected without crash | Pass if engine returns error code `-ERR` without floating-point exception or crash. |
| **TC-RING-01** | Ring Eviction | Saturation & $O(1)$ circular wrap | Push $N+1$, $2N$, $10N$ items into ring capacity $N$ | `ring->count == N`, `total_appended == 10N`, oldest items overwritten | Pass if `ring->head == (total % N)` and head pointer remains consistent. |
| **TC-RING-02** | Ring Eviction | Zero dangling references on wrap | Wrap buffer 1,000 times, query overwritten IDs | Overwritten IDs returned as invalid/not found | Pass if no stale or corrupt pointers are accessible after eviction. |
| **TC-TTL-01** | TTL Mechanics | Passive lazy deletion vs. active sweep | Set key with $50\text{ ms}$ TTL. Query at $T=10\text{ ms}$ vs $T=100\text{ ms}$ | `GET` returns vector at $10\text{ ms}$, returns `NULL` at $100\text{ ms}$ | Pass if passive access evicts entry and `expired_count` increments. |
| **TC-MEM-01** | Memory & Alignment | 64-byte SIMD cache-line alignment | Allocate 10,000 vectors via `palloc` | `(uintptr_t)vec->data % 64 == 0` | Pass if 100% of vector data pointers satisfy 64-byte alignment assertion. |
| **TC-MEM-02** | Memory & Safety | ASan & Valgrind leak audit | Run 100,000 allocation & eviction cycles under ASan | Zero memory leaks, zero heap fragmentation | Pass if AddressSanitizer exits with `0` errors. |
| **TC-CONC-01** | Concurrency | Data race detection (TSan) | 16 threads concurrently running `PUSH`, `GET`, `SEARCH`, `PURGE` | Zero data races on shared `rwlock` & ring pointers | Pass if ThreadSanitizer flags zero data races during high-concurrency execution. |
| **TC-NET-01** | Network Chaos | TCP stream fragmentation | Fragment TCP command lines across multiple TCP segments | Command buffer correctly reassembles payload | Pass if TCP socket server responds with valid response without corruption. |
| **TC-NET-02** | Network Chaos | Malformed commands & buffer overflow | 10MB payload, invalid UTF-8, oversized $k > N$ | Buffer bounds enforced, response `-ERR syntax_error` | Pass if connection handles bad input gracefully without buffer overflow. |
| **TC-PERF-01** | Performance | Latency percentiles under load | 10,000 Cosine distance queries over 10,000 vectors | $p50 < 2\text{ ms}$, $p99 < 5\text{ ms}$ | Pass if tail latency remains bounded under 100% load. |

---

## 2. CI/CD Sanitizer & Automation Matrix

### Recommended Compiler Flags
```bash
# Production release build
gcc -O3 -march=native -mavx2 -mfma -Wall -Wextra -D_GNU_SOURCE

# ASan + UBSan Debug Build
gcc -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -march=native -mavx2 -mfma -D_GNU_SOURCE

# ThreadSanitizer Build
gcc -g -O2 -fsanitize=thread -march=native -mavx2 -mfma -D_GNU_SOURCE
```

### Automation Execution Targets
- `make test` : Runs standard functional & precision unit tests.
- `make test-asan` : Compiles with ASan/UBSan and verifies zero memory leaks.
- `make test-tsan` : Compiles with ThreadSanitizer and verifies zero data races.
- `make bench` : Executes SIMD vector search and throughput benchmarks.
