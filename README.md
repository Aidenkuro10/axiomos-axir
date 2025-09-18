AXIR — A Universal IR for GPU/AI Code

AXIR is a proof-of-concept universal intermediate representation (IR) designed to make GPU kernels portable across ecosystems.

Write your code in CUDA (NVIDIA), HIP (AMD), OpenCL (Khronos), or SYCL (Intel/oneAPI).
Translate it once into AXIR. Run it anywhere.

Why AXIR?

Why not just use OpenCL?
OpenCL asks you to rewrite kernels in its C-like language. AXIR is different: you keep writing in CUDA, HIP, OpenCL, or SYCL. AXIR acts as a translator layer so your existing kernels become portable without rewrites.

Related work (great, but different focus):

LLVM IR / MLIR — powerful compiler IRs, but not a minimal GPU portability pivot.

Triton — high-level DSL for kernels, vendor-specific runtime.

oneAPI / SYCL — tied to one ecosystem.

👉 AXIR’s goal: a minimal pivot IR across all vendor ecosystems.

Current Status (PoC v0.7)

✅ Frontends: CUDA, HIP, OpenCL, SYCL
✅ Backends: CPU (NumPy), OpenCL (real GPU), GPU-stub (simulation; optional)
✅ Kernels supported (PoC):

vector_add

saxpy (C = α·A + B)

reduce_sum (sum(A))

matmul (matrix multiplication)

conv1d (basic prototype)

Compatibility Matrix (PoC)
Frontend ↓	CPU backend	OpenCL backend	GPU-stub
CUDA	✅	✅	✅
HIP	✅	✅	✅
OpenCL	✅	✅	✅
SYCL	✅	✅	✅

All combinations above run and match numerically on small demos (see cli/verify_axir.py for cross-backend checks).

Requirements

Python 3.10+

NumPy

PyOpenCL (for GPU runs; CPU-only runs don’t need it)

OpenCL runtime/driver from your vendor (Intel / AMD / NVIDIA).
If unsure, install an Intel CPU OpenCL runtime as a fallback.

Quick Setup (5 min)
# 1) Clone
git clone https://github.com/<your-username>/axiomos.git
cd axiomos

# 2) Create a virtualenv
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# 3) Install deps
pip install -U pip
pip install -r requirements.txt  # or at least: numpy pyopencl


Tip: If you only want CPU runs, numpy is enough. Install pyopencl + a vendor runtime when you’re ready to try GPU.

Sanity Check
# List detected devices (CPU/OpenCL/CUDA if present)
python -m cli.axirc device-list


You should see at least one OpenCL platform/device. If not, you can still run CPU-only demos.

First Run (works on any machine)
# Prebaked AXIR (CUDA → AXIR) on CPU
python -m cli.axirc demo-prebaked --kernel vector_add --frontend cuda

# Or run the prebaked AXIR file explicitly:
python -m cli.axirc run --in build/vector_add_from_cuda.axir.json --backend cpu


Expected (example):

---- RESULT ----
SHAPES     : CPU(16,)
CPU(head): [0., 3., 6., 9., 12., 15., 18., 21.]

GPU Run (OpenCL)
python -m cli.axirc run --in build/vector_add_from_cuda.axir.json --backend opencl


You should see the same numeric values as the CPU backend.

To verify CPU vs GPU formally and (optionally) time them:

python -m cli.verify_axir build/vector_add_from_cuda.axir.json --buffer auto \
  --backend-a cpu --backend-b opencl --time --warmup 2 --repeat 5


This prints shapes, head values, a strict np.allclose check, and median wall-clock timings (includes subprocess & I/O to keep it simple and comparable).

Quick Demos (live translation)

Translate from a frontend and run through AXIR:

# Live translation → AXIR → run (shows AXIR snippet + timings)
python -m cli.axirc demo --kernel vector_add --frontend cuda
python -m cli.axirc demo --kernel reduce_sum --frontend hip
python -m cli.axirc demo --kernel vector_add --frontend opencl
python -m cli.axirc demo --kernel vector_add --frontend sycl


Prebaked AXIR fast path:

python -m cli.axirc demo-prebaked --kernel vector_add  --frontend cuda
python -m cli.axirc demo-prebaked --kernel vector_add  --frontend hip
python -m cli.axirc demo-prebaked --kernel saxpy       --frontend hip
python -m cli.axirc demo-prebaked --kernel reduce_sum  --frontend hip
python -m cli.axirc demo-prebaked --kernel matmul      --frontend axir
python -m cli.axirc demo-prebaked --kernel conv1d      --frontend axir


Unified CLI (early WIP):

axirc translate --in demos/saxpy/saxpy.cu --lang cuda --out build/saxpy.axir.json
axirc run       --in build/saxpy.axir.json            --backend cpu
axirc run       --in build/saxpy.axir.json            --backend opencl

Cross-Backend Verification (CLI)

Compare a single buffer between two backends:

# List buffers contained in an AXIR JSON
python -m cli.verify_axir build/saxpy_from_cuda.axir.json --list-buffers

# Compare (default: CPU vs OpenCL)
python -m cli.verify_axir build/matmul_from_hip.axir.json --buffer hC
# Or pick backends explicitly
python -m cli.verify_axir build/matmul_from_hip.axir.json --buffer hC --backend-a cpu --backend-b cuda


Smoke-test a set of AXIR files and write a report:

python -m cli.smoke_axir --dir build --buffer auto --backend-a cpu --backend-b opencl \
  --report-md build/verify_report.md --report-csv build/verify_report.csv

How It Works (high level)

Frontends (CUDA/HIP/OpenCL/SYCL) parse or stub the kernel launch and memory ops and lower them to AXIR JSON:

DeviceSelect, DeviceMalloc, Memcpy(H2D/D2H), KernelLaunch, etc.

Backends read AXIR and execute:

CPU backend uses NumPy as a reference interpreter.

OpenCL backend builds simple OpenCL kernels (vector_add, saxpy, reduce_sum, matmul) and runs them on the selected device.

verify_axir runs two backends, dumps a chosen buffer (e.g., hC, hOut) to .npy, and checks np.allclose.

Buffer naming convention (PoC):
Host buffers hA/hB/hC/..., device buffers dA/dB/dC/....
Frontends/backends are tolerant and try to infer sizes when metadata is missing.

Benchmarks (illustrative)

AXIR shows that the same kernels translated from CUDA/HIP can run on multiple backends and match numerically. Timing shown by verify_axir --time is end-to-end (spawn + I/O) for simplicity and fairness in the PoC.

Expect numbers to vary by machine and drivers. The point of the PoC is portability + correctness, not absolute performance tuning (yet).

Troubleshooting

ImportError: pyopencl → pip install pyopencl

“No OpenCL platforms found” → install a vendor OpenCL runtime/driver (Intel/AMD/NVIDIA).

Windows PowerShell: quote paths with spaces, e.g. "build\vector_add_from_cuda.axir.json".

Warnings like [WARN] Backend ... introuvable → harmless in PoC; it just means that backend script isn’t installed on your machine.

Roadmap

0–6 weeks (Core)

AXIR v0.1 spec (types, memory, sync, intrinsics)

Conformance tests + compatibility matrix

Benchmark harness (CPU vs OpenCL)

7–18 weeks (GPU)

Additional real GPU backends (CUDA/PTX, ROCm)

Make ≥2 kernels run <2× slower than native on at least one GPU

Notebook demo “multi-frontend → real GPU”

19–28 weeks (Portability)

SPIR-V backend (Vulkan/OpenCL path)

Cross-vendor benchmarks at scale

Contributing

Issues and PRs are welcome. This is early research software—clear bug reports and small, focused PRs help the most.

License

TBD (PoC). Replace with MIT/Apache-2.0 once finalized.

 Contact

**Pierre Seck**  
✉️ Email: pierre.seck@unine.ch
LinkedIn: www.linkedin.com/in/pierre-seck-0b9391381


