AXIR — A Universal IR for GPU/AI Code (PoC)

Run unmodified CUDA/HIP kernels on multiple backends.
AXIR is a minimal intermediate representation (IR) and runtime harness that lets you translate once and execute across devices.

Frontends (today): CUDA, HIP (live); OpenCL, SYCL (prebaked AXIR demos)

Backends (today): CPU (NumPy, reference) and OpenCL GPU (tested on Intel iGPU)

Kernels covered: vector_add, saxpy, reduce_sum, matmul

Verification: buffer-level CPU ↔ GPU checks with configurable tolerances and timing

Goal: a neutral compute layer for AI — a lightweight pivot IR + runtimes that break vendor lock-in.

Why AXIR?

Existing paths are powerful but fragmented:

Write once in vendor languages (CUDA/HIP/OpenCL/SYCL) → translate to AXIR → run on different backends.

Unlike “just use OpenCL/SYCL”, AXIR lets teams keep their existing CUDA/HIP kernels and run them elsewhere with minimal changes.

Compared to general IRs (LLVM/MLIR) or DSLs (Triton), AXIR is a tiny, pragmatic pivot focused on portability + runnable demos.

What works today (PoC)

✅ Live frontends: CUDA, HIP → AXIR JSON

✅ Prebaked AXIR demos: from CUDA/HIP/OpenCL/SYCL for quick runs

✅ Backends:

CPU (NumPy): reference execution

OpenCL GPU: real device execution (uses PyOpenCL; works on Intel iGPU; falls back to first OpenCL device)

✅ Kernels: vector_add, saxpy, reduce_sum, matmul

✅ Verification & timing: compare host buffers (e.g., hC, hOut) across backends; --time --warmup --repeat for stable medians

What this PoC is not (yet): production codegen (PTX/SPIR-V), full CUDA runtime emulation, or a wide kernel library.

Requirements

Python 3.10+

numpy, pyopencl (pip install numpy pyopencl)

An OpenCL runtime/driver (GPU or CPU OpenCL device).

Windows/Linux/macOS ok; PoC validated on Windows with Intel iGPU.

Quickstart
# List detected devices (CPU/OpenCL/CUDA if present)
python -m cli.axirc device-list

Live demo (translate → run)
# Generate AXIR from a toy frontend and run backends
python -m cli.axirc demo --kernel vector_add --frontend cuda --summary --with-opencl
python -m cli.axirc demo --kernel reduce_sum --frontend hip   --summary --with-opencl

Prebaked demos (fast path)
# Runs AXIR JSONs included in build/ (no translator needed)
python -m cli.axirc demo-prebaked --kernel vector_add --frontend cuda
python -m cli.axirc demo-prebaked --kernel saxpy      --frontend hip
python -m cli.axirc demo-prebaked --kernel reduce_sum --frontend hip
python -m cli.axirc demo-prebaked --kernel matmul     --frontend axir

Run a specific AXIR on a backend
python -m cli.axirc run --in build/vector_add_from_cuda.axir.json --backend cpu
python -m cli.axirc run --in build/vector_add_from_cuda.axir.json --backend opencl

Verify correctness (CPU vs GPU) + timing

We provide a harness to dump a chosen buffer from two backends, compare numerically, and (optionally) time multiple runs:

# List candidate buffers inside the AXIR file
python -m cli.verify_axir build/saxpy_from_cuda.axir.json --list-buffers

# Auto-pick a likely output buffer and verify CPU vs OpenCL
python -m cli.verify_axir build/saxpy_from_cuda.axir.json --buffer auto \
  --backend-a cpu --backend-b opencl

# Add timing with warmup and repeats (median)
python -m cli.verify_axir build/vector_add_from_opencl.axir.json --buffer auto \
  --backend-a cpu --backend-b opencl \
  --time --warmup 2 --repeat 5


Sample output:

---- RESULT ----
SHAPES     : CPU(16,) vs OPENCL(16,)
max_abs_err: 0.0
ALLCLOSE   : True (atol=1e-06, rtol=0.0)
CPU(head):    [0., 3., 6., 9., 12., 15., 18., 21.]
OPENCL(head): [0., 3., 6., 9., 12., 15., 18., 21.]

---- TIMING ----
Warmup: 2, Repeats: 5 (median)
CPU     : 231.99 ms
OPENCL  : 471.18 ms


Notes:

--buffer auto picks a likely output (e.g., hC, hOut).

For large matrices you can relax tolerances: --atol 1e-3 --rtol 5e-6.

Times include process + I/O overhead (good for gross comparisons, not micro-benchmarks).

Included demos (prebaked AXIR)

The repo ships AXIR JSONs such as:

vector_add_from_{cuda,hip,opencl,sycl}.axir.json

saxpy_from_{cuda,hip}.axir.json

reduce_sum_from_{cuda,hip}.axir.json

matmul_from_hip.axir.json

All included demos pass CPU ↔ OpenCL verification on our test machine.

Roadmap

0–6 weeks (Core):

AXIR v0.1 spec (types/memory/sync/intrinsics), conformance tests

Better codegen paths in frontends; CLI polish

7–18 weeks (GPU):

First real codegen backend (PTX or SPIR-V/Vulkan)

Early perf targets (<2× from native) on a few kernels

Subgraph demo on a real model fragment

19–28 weeks (Portability):

Additional backends (ROCm, Vulkan/SPIR-V path)

Cross-vendor perf + correctness matrix

Status & Disclaimer

This is an early PoC intended to demonstrate feasibility: translate unmodified CUDA/HIP to a small IR and run it across backends, with automated verification. Expect rough edges; contributions and issues are welcome.
