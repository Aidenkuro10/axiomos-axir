# AXIR — A Universal IR for GPU/AI Code

AXIR is a proof-of-concept **universal intermediate representation (IR)**  
designed to make GPU kernels portable across ecosystems.

Write your code in **CUDA (NVIDIA)**, **HIP (AMD)**, **OpenCL (Khronos)**, or **SYCL (Intel/oneAPI)**.  
Translate it once into AXIR. Run it anywhere.

---

## Why AXIR?

**Why not just use OpenCL?**  
OpenCL forces developers to rewrite code in its own C-like kernel language.  
AXIR is different: you keep writing in CUDA, HIP, OpenCL, or SYCL.  
AXIR acts as a translator layer, making your code portable *without rewrites*.

Other projects address parts of this challenge:
- **LLVM IR / MLIR** — powerful compiler IRs, but not designed as a lightweight GPU pivot.  
- **Triton** — high-level DSL for kernels, but vendor-specific.  
- **oneAPI / SYCL** — Intel ecosystem, not universal.  

👉 AXIR focuses on one simple goal: **a minimal pivot IR across all vendor ecosystems**.

---

## Current Status (PoC v0.6)

✅ **Frontends**: CUDA, HIP, OpenCL, SYCL  
✅ **Backends**: CPU (NumPy), GPU stub (simulation, CuPy if available)  
✅ **Kernels supported**:
- `vector_add`
- `saxpy` (C = α*A + B)
- `reduce_sum` (sum(A))
- `matmul` (matrix multiplication)
- `conv1d` (1D convolution, basic prototype)

---

## Compatibility Matrix (PoC)

| Frontend ↓ | CPU backend | GPU-stub backend |
|------------|-------------|------------------|
| CUDA       | ✅ (5/5)    | ✅ (5/5)         |
| HIP        | ✅ (5/5)    | ✅ (5/5)         |
| OpenCL     | ✅ (5/5)    | ✅ (5/5)         |
| SYCL       | ✅ (5/5)    | ✅ (5/5)         |

All **20 combinations** (4 frontends × 5 kernels) successfully translate into AXIR and run on CPU and GPU-stub, producing the expected results.

---

## Demo (GUI)

Launch the one-button demo:

```bash
# Windows
py -3 axir_demo_gui.py
# Linux/macOS
python3 axir_demo_gui.py
You’ll see a window. Click ▶ Run Full AXIR Demo.
It will sequentially run:

nginx
Copier le code
CUDA / HIP / OpenCL / SYCL  →  AXIR  →  CPU + GPU-stub
Expected outputs (examples):

vector_add → hC = [0, 3, 6, 9, 12]

saxpy → hC = [0, 4, 8, 12, 16]

reduce_sum → hOut = 120.0

matmul → hC[0:2,0:2] = [[19, 22], [43, 50]]

conv1d → hOut[0:2,0:2] = [[4, 8], [12, 16]]

Demo (CLI)
Step-by-step execution:

# Example: CUDA saxpy → AXIR → CPU
python frontends/cuda_frontend.py demos/saxpy/saxpy.cu -o demos/saxpy/saxpy_from_cuda.axir.json
python backends/cpu_numpy_backend.py demos/saxpy/saxpy_from_cuda.axir.json --summary
Unified CLI (in progress):

axirc translate --in demos/saxpy/saxpy.cu --lang cuda --out build/saxpy.axir.json
axirc run --in build/saxpy.axir.json --backend cpu
axirc run --in build/saxpy.axir.json --backend gpu-stub
Quick Demos (local, no Colab)
Live translation → AXIR → run (shows AXIR snippet + timings)

python cli/axirc.py demo --kernel vector_add --frontend cuda
python cli/axirc.py demo --kernel reduce_sum --frontend hip
python cli/axirc.py demo --kernel vector_add --frontend opencl
python cli/axirc.py demo --kernel vector_add --frontend sycl
Prebaked AXIR → run (fast path)

python cli/axirc.py demo-prebaked --kernel vector_add --frontend cuda
python cli/axirc.py demo-prebaked --kernel vector_add --frontend hip
python cli/axirc.py demo-prebaked --kernel vector_add --frontend opencl
python cli/axirc.py demo-prebaked --kernel vector_add --frontend sycl
python cli/axirc.py demo-prebaked --kernel saxpy      --frontend hip
python cli/axirc.py demo-prebaked --kernel reduce_sum --frontend hip
python cli/axirc.py demo-prebaked --kernel matmul     --frontend axir
python cli/axirc.py demo-prebaked --kernel conv1d     --frontend axir
What you see:

Generated AXIR JSON snippet (proves the IR pivot)

CPU & GPU-stub executions with numeric outputs

Per-stage timings (measured, not mocked)

Benchmarks
AXIR demonstrates that the same computation kernels (CUDA/HIP) can be translated into a unified IR and executed across multiple backends, producing correct results with reproducible timings.

We evaluated AXIR on multiple kernels translated from CUDA and HIP, executed on two backends:

CPU backend: NumPy-based interpreter (reference execution).

GPU-stub backend: GPU API simulation (uses CuPy if available, otherwise falls back to NumPy).

Each kernel was run 7 times to capture variability.
Results are reported as mean ± standard deviation (in milliseconds).

Results (7 runs)
<!-- auto-generated via `python cli/axirc.py bench --runs 7` -->
Kernel	Frontend	CPU (ms)	GPU-stub (ms)
vector_add	CUDA	162.6 ± 8.8	 160.7 ± 10.1
saxpy	HIP	156.7 ± 8.9 	158.1 ± 7.5
reduce_sum	HIP	149.1 ± 4.8 	149.2 ± 2.8

Visualization
The bar plot shows average execution time per kernel, with error bars representing standard deviation.



Roadmap
0–6 weeks (Core)

AXIR v0.1 spec (types, memory, sync, intrinsics)

Conformance tests + compatibility matrix

Benchmark harness (CPU vs GPU-stub)

7–18 weeks (GPU)

First real GPU backend (CUDA/PTX or ROCm)

At least 2 kernels run <2× slower than native

Notebook demo “multi-frontend → GPU”

19–28 weeks (Portability)

Backend SPIR-V (Vulkan/OpenCL path)

Cross-vendor benchmarks

Landing page + docs

License
Apache-2.0 (with CLA for contributions).


