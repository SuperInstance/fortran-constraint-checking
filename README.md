# fortran-constraint-checking

**High-performance multi-language constraint checking** — Fortran core with Python, Rust, and C bindings, optimized for AMD Zen 5 / AVX-512 auto-vectorization.

## Why Fortran?

Fortran auto-vectorizes better than C for constraint checking because:

1. **No pointer aliasing by default** — the compiler can freely reorder and SIMD-ify array operations
2. **Array-first semantics** — whole-array operations map directly to SIMD instructions
3. **Intrinsics like `count()`** — compiler emits `vpcmpq + vpopcntdq` without manual intrinsics
4. **Proven in HPC** — decades of numerical optimization in compilers

On AMD Zen 5 with AVX-512, the Fortran constraint checker achieves:

| Operation | Throughput | SIMD Width |
|-----------|-----------|------------|
| Range check (f64) | ~8 elements/cycle | 512-bit (8 doubles) |
| Bitmask AND (i64) | ~8 elements/cycle | 512-bit (8 int64s) |
| Count in range | ~8 elements/cycle + popcnt | 512-bit + vpopcntdq |
| Multi-constraint | ~8 elements/cycle | 512-bit |

## Quick Start

### Build the Fortran Library

```bash
# Compile the Fortran module
gfortran -O3 -march=znver5 -ffast-math -c constraint_checker.f90

# Create a shared library
gfortran -O3 -march=znver5 -ffast-math -shared -fPIC \
  -o libconstraint_checker.so constraint_checker.f90

# Or create a static archive
ar rcs libconstraint_checker.a constraint_checker.o
```

### Python Usage

```python
from constraint_checker import ConstraintChecker
import numpy as np

cc = ConstraintChecker()

# Range check: which values fall in [0.0, 1.0]?
values = np.array([0.1, 0.5, 1.2, -0.3, 0.8])
mask = cc.check_range(values, lo=0.0, hi=1.0)
# mask = [True, True, False, False, True]

# Count values in range (no mask allocation)
count = cc.count_in_range(values, lo=0.0, hi=1.0)
# count = 3

# Bitmask domain check
domains = np.array([0xFF00, 0x0FF0, 0x00FF], dtype=np.int64)
result = cc.check_bitmask(domains, mask_bits=0xF0F0)
# result = [0xF000, 0x00F0, 0x0000]

# Multi-constraint check (multiple [lo, hi] ranges)
lo = np.array([0.0, 0.2, 0.4])
hi = np.array([0.3, 0.6, 0.8])
result = cc.check_multi_constraint(values, lo_bounds=lo, hi_bounds=hi)
```

### C Usage

```c
#include "constraint_checker.h"
#include <stdio.h>

int main() {
    double values[] = {0.1, 0.5, 1.2, -0.3, 0.8};
    bool mask[5];
    
    check_range_f64(values, 5, 0.0, 1.0, mask);
    
    for (int i = 0; i < 5; i++) {
        printf("values[%d] = %.1f → %s\n", i, values[i], 
               mask[i] ? "IN RANGE" : "OUT OF RANGE");
    }
    return 0;
}
```

Compile and link:
```bash
gcc -O3 -o my_check main.c constraint_checker.o -lgfortran -lm
```

### Rust Usage

```rust
use constraint_checker::{check_range, count_in_range, check_bitmask};

fn main() {
    let values = vec![0.1, 0.5, 1.2, -0.3, 0.8];
    
    let mask = check_range(&values, 0.0, 1.0);
    println!("Mask: {:?}", mask);
    
    let count = count_in_range(&values, 0.0, 1.0);
    println!("Count in range: {}", count);
    
    let domains = vec![0xFF00i64, 0x0FF0, 0x00FF];
    let result = check_bitmask(&domains, 0xF0F0);
    println!("Bitmask result: {:?}", result);
}
```

## API Reference

### Fortran Module: `constraint_checker`

#### Subroutines

##### `check_range(values, lo, hi, mask)`
Check if each element of `values` falls within `[lo, hi]`.

- `values(:)` — real(dp), intent(in) — input array
- `lo` — real(dp), intent(in) — lower bound
- `hi` — real(dp), intent(in) — upper bound
- `mask(:)` — logical, intent(out) — result mask

**Auto-vectorizes to**: `vcmppd + vandpd` (AVX-512)

##### `check_bitmask(domains, mask_bits, result)`
Bitwise AND of each domain with mask_bits.

- `domains(:)` — integer(i64), intent(in) — domain descriptors
- `mask_bits` — integer(i64), intent(in) — bits to test
- `result(:)` — integer(i64), intent(out) — AND results

**Auto-vectorizes to**: `vpandq` (AVX-512)

##### `check_multi_constraint(values, lo, hi, result)`
Check values against multiple [lo, hi] range pairs simultaneously.

- `values(:)` — real(dp), intent(in)
- `lo(:)` — real(dp), intent(in) — lower bounds (one per constraint)
- `hi(:)` — real(dp), intent(in) — upper bounds (one per constraint)
- `result(:)` — logical, intent(out) — combined satisfaction mask

##### `domain_intersection(domains_a, domains_b, result)`
Compute the intersection of two domain sets via bitwise AND.

##### `domain_union(domains_a, domains_b, result)`
Compute the union of two domain sets via bitwise OR.

#### Functions

##### `count_in_range(values, lo, hi) → integer`
Count values in `[lo, hi]` without allocating a mask.

**Auto-vectorizes to**: `vpcmpq + vpopcntdq` (AVX-512)

### C Bindings

All Fortran functions have C-compatible bindings via `iso_c_binding`:

| Fortran | C Function |
|---------|-----------|
| `check_range` | `check_range_f64` |
| `count_in_range` | `count_in_range_f64` |
| `check_bitmask` | `check_bitmask_i64` |
| `check_multi_constraint` | `check_multi_f64` |

### Python API

```python
class ConstraintChecker:
    def __init__(self, lib_path: str = None)
    def check_range(self, values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray
    def count_in_range(self, values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> int
    def check_bitmask(self, domains: np.ndarray, mask_bits: int) -> np.ndarray
    def check_multi_constraint(self, values: np.ndarray, lo_bounds: np.ndarray, hi_bounds: np.ndarray) -> np.ndarray
```

### Rust API

```rust
pub fn check_range(values: &[f64], lo: f64, hi: f64) -> Vec<bool>
pub fn count_in_range(values: &[f64], lo: f64, hi: f64) -> usize
pub fn check_bitmask(domains: &[i64], mask_bits: i64) -> Vec<i64>
pub fn check_multi_constraint(values: &[f64], lo: &[f64], hi: &[f64]) -> Vec<bool>
```

## File Structure

```
├── constraint_checker.f90    # Fortran module — core implementation
├── constraint_checker.h      # C header for FFI
├── constraint_checker.py     # Python bindings via ctypes
├── constraint_checker.rs     # Rust FFI bindings
├── constraint_checker.mod    # Compiled Fortran module file
├── constraint_checker_c.mod  # Compiled Fortran C-binding module
├── bench_constraint          # Pre-built benchmark binary
└── report.md                 # Full research report with benchmarks
```

## Performance Report

The included `report.md` contains a detailed analysis covering:

- **Fortran vs C for SIMD**: Why Fortran auto-vectorizes better (alias rules, array semantics)
- **Auto-vectorization analysis**: Assembly output showing AVX-512 instruction mapping
- **Benchmarks**: Throughput measurements on AMD Zen 5
- **Scaling characteristics**: Performance across array sizes from 64 to 10M elements
- **Multi-constraint optimization**: How compound checks maintain vectorization

### Key Benchmark Results

| Array Size | Range Check (ns/elem) | Count (ns/elem) | Bitmask (ns/elem) |
|-----------|----------------------|-----------------|-------------------|
| 64 | 0.12 | 0.11 | 0.08 |
| 1K | 0.09 | 0.08 | 0.06 |
| 1M | 0.08 | 0.07 | 0.05 |
| 10M | 0.08 | 0.07 | 0.05 |

## Building the Benchmark

```bash
# Build the benchmark binary
gfortran -O3 -march=znver5 -ffast-math -o bench_constraint constraint_checker.f90 bench_main.f90

# Run
./bench_constraint
```

## Use Cases

### Constraint Satisfaction Problems (CSP)

```python
from constraint_checker import ConstraintChecker
import numpy as np

cc = ConstraintChecker()

# Check 100K random points against constraint bounds
points = np.random.randn(100000)
valid = cc.check_range(points, lo=-2.0, hi=2.0)
valid_count = cc.count_in_range(points, lo=-2.0, hi=2.0)
print(f"{valid_count} of {len(points)} points satisfy constraints")
```

### Bitmask Domain Operations

```python
# Each int64 represents a 64-dimension constraint domain
domains = np.random.randint(0, 2**64, size=10000, dtype=np.int64)

# Find domains that have specific bits set
mask = 0xAAAAAAAA55555555  # alternating bit pattern
result = cc.check_bitmask(domains, mask)

# Domain intersection
domains_a = np.array([0xFF, 0xAA, 0x55], dtype=np.int64)
domains_b = np.array([0xF0, 0x0F, 0xFF], dtype=np.int64)
# intersection = a & b
```

### Multi-Constraint Validation

```python
# Validate against multiple constraint bounds simultaneously
values = np.random.randn(50000)

# Three constraint ranges
lo = np.array([-1.0, 0.0, -0.5])
hi = np.array([1.0, 2.0, 0.5])

result = cc.check_multi_constraint(values, lo_bounds=lo, hi_bounds=hi)
# result[i] = True if values[i] satisfies ALL three constraints
```

## Compiler Requirements

- **gfortran** ≥ 12 (for full AVX-512 / Zen 5 support)
- **gcc** ≥ 12 (for C bindings)
- **rustc** ≥ 1.70 (for Rust bindings)
- **Python** ≥ 3.10 (for Python bindings)

### Recommended Flags

```bash
-O3 -march=znver5 -ffast-math
```

For debugging (no vectorization):
```bash
-O0 -fno-vectorize
```

## Related Projects

| Repository | Description |
|-----------|-------------|
| [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) | Core constraint theory mathematics |
| [constraint-viz](https://github.com/SuperInstance/constraint-viz) | Multi-scale constraint visualization oscilloscope |
| [constraint-theory-py](https://github.com/SuperInstance/constraint-theory-py) | Python constraint theory library |
| [deadband-rs](https://github.com/SuperInstance/deadband-rs) | Rust deadband computation |
| [deadband-zig](https://github.com/SuperInstance/deadband-zig) | Zig deadband implementation |
| [cuda-constraint-checker](https://github.com/SuperInstance/cuda-constraint-checker) | GPU-accelerated constraint checking |
| [flux-check](https://github.com/SuperInstance/flux-check) | Flux language constraint checker |

## Citation

```bibtex
@software{fortran_constraint_checking_2026,
  title = {fortran-constraint-checking: High-Performance Multi-Language Constraint Checking},
  author = {Forgemaster Research},
  year = {2026},
  url = {https://github.com/SuperInstance/fortran-constraint-checking}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.

---

Part of the [SuperInstance](https://github.com/SuperInstance) constraint theory ecosystem.
