 AXIR — A Universal IR for GPU/AI Code

AXIR is a proof-of-concept universal intermediate representation (IR) that makes GPU code portable across ecosystems.

Write in **CUDA (NVIDIA)**, **HIP (AMD)**, **OpenCL (Khronos)**, or **SYCL (Intel/oneAPI)** — translate into AXIR — and run it anywhere (currently: on CPU and GPU stub backends).

---

## Why not just OpenCL?
- OpenCL forces developers to rewrite code in its own C-like kernel language.  
- AXIR is different: you keep writing in CUDA/HIP/OpenCL/SYCL.  
- AXIR acts as a translator layer, making your code portable without rewrites.  

Other projects (LLVM IR, MLIR, Triton, oneAPI) tackle parts of the problem, but none offer a **simple pivot IR** focused on portability across **all vendor ecosystems**.

---

## Current Status (PoC v0.6)

✅ **Frontends**: CUDA, HIP, OpenCL, SYCL  
✅ **Backends**: CPU (NumPy), GPU stub (simulation)  
✅ **Kernels supported**:  
- `vector_add`  
- `saxpy` (C = α*A + B)  
- `reduce_sum` (sum(A))  
- `matmul` (matrix multiplication)  
- `conv` (2D convolution, basic)  

### Compatibility matrix (PoC)
| Frontend ↓ | CPU backend | GPU stub backend |
|------------|-------------|------------------|
| CUDA       | ✅ (5/5)    | ✅ (5/5)         |
| HIP        | ✅ (5/5)    | ✅ (5/5)         |
| OpenCL     | ✅ (5/5)    | ✅ (5/5)         |
| SYCL       | ✅ (5/5)    | ✅ (5/5)         |

All 20 combinations (4 frontends × 5 kernels) translate into AXIR and run successfully on CPU and GPU stub, producing the expected results.

---

## Demo (GUI)
Launch the one-button demo:

```bash
py -3 axir_demo_gui.py

You’ll see a window. Click ▶ Run Full AXIR Demo.
It will sequentially run:

CUDA / HIP / OpenCL / SYCL  →  AXIR  →  CPU + GPU stub
Expected outputs (examples):

vector_add → hC = [0, 3, 6, 9, 12]

saxpy → hC = [0, 4, 8, 12, 16]

reduce_sum → hOut = 120.0

matmul → hC[0:2,0:2] = [[19, 22], [43, 50]]

conv → hOut[0:2,0:2] = [[4, 8], [12, 16]]

Demo (CLI)

Step-by-step execution:

# Example: CUDA saxpy → AXIR → CPU
py -3 frontends/cuda_frontend.py demos/saxpy/saxpy.cu -o demos/saxpy/saxpy_from_cuda.axir.json
py -3 backends/cpu_numpy_backend.py demos/saxpy/saxpy_from_cuda.axir.json --summary


We are working towards a unified CLI:

axirc translate --in demos/saxpy/saxpy.cu --lang cuda --out build/saxpy.axir.json
axirc run --in build/saxpy.axir.json --backend cpu
axirc run --in build/saxpy.axir.json --backend gpu-stub

## Quick demos (local, no Colab)

# Live translation → AXIR → run (shows AXIR snippet + timings)
python cli/axirc.py demo --kernel vector_add --frontend cuda
python cli/axirc.py demo --kernel reduce_sum --frontend hip
python cli/axirc.py demo --kernel vector_add --frontend opencl
python cli/axirc.py demo --kernel vector_add --frontend sycl

# Prebaked AXIR → run (fast path)
python cli/axirc.py demo-prebaked --kernel vector_add --frontend cuda
python cli/axirc.py demo-prebaked --kernel vector_add --frontend hip
python cli/axirc.py demo-prebaked --kernel vector_add --frontend opencl
python cli/axirc.py demo-prebaked --kernel vector_add --frontend sycl
python cli/axirc.py demo-prebaked --kernel saxpy      --frontend hip
python cli/axirc.py demo-prebaked --kernel reduce_sum --frontend hip
python cli/axirc.py demo-prebaked --kernel matmul     --frontend axir
python cli/axirc.py demo-prebaked --kernel conv1d     --frontend axir

**What you see:** 
- Generated AXIR JSON snippet (proves the IR pivot)
- CPU & GPU-stub executions with numeric outputs
- Per-stage timings (not just prints)


Roadmap

0–6 weeks (Core)

AXIR v0.1 spec (types, memory, sync, intrinsics)

Conformance tests + compatibility matrix

Benchmark harness (CPU vs GPU stub)

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
