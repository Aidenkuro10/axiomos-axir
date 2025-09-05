# AXIR — A Universal IR for GPU/AI Code

**AXIR** is a proof-of-concept universal **intermediate representation (IR)** that makes GPU code portable across ecosystems.  

Write in **CUDA (NVIDIA)**, **HIP (AMD)**, **OpenCL (Khronos)**, or **SYCL (Intel/oneAPI)** — translate into **AXIR** — and run it anywhere (here: on a CPU backend).

---

##  Why not just OpenCL?
- OpenCL forces developers to rewrite code in its own language.  
- AXIR is different: you keep writing in CUDA/HIP/OpenCL/SYCL.  
- AXIR acts as a **translator layer**, making your code portable without rewrites.

---

## Current status (PoC v0.5)
- **4 frontends:** CUDA, HIP, OpenCL, SYCL  
- **1 backend:** CPU (NumPy)  
- **3 kernels supported:**  
  - `vector_add`  
  - `saxpy` (`C = α*A + B`)  
  - `reduce_sum` (`sum(A)`)  

All 12 combinations (4 languages × 3 kernels) translate into AXIR and run successfully on CPU, producing the expected results.

---

##  Demo (GUI)
Launch the one-button demo:

```bash
py -3 axir_demo_gui.py
You’ll see a window. Click ▶ Run Full AXIR Demo.
It will sequentially run:
CUDA / HIP / OpenCL / SYCL  →  AXIR  →  CPU
Expected outputs:
vector_add → hC = [0, 3, 6, 9, 12]
saxpy → hC = [0, 4, 8, 12, 16]
reduce_sum → hOut = 120.0

Demo (CLI)
You can also run step by step:
# Example: CUDA saxpy → AXIR → CPU
py -3 frontends/cuda_frontend.py demos/saxpy/saxpy.cu -o demos/saxpy/saxpy_from_cuda.axir.json
py -3 backends/cpu_numpy_backend.py demos/saxpy/saxpy_from_cuda.axir.json --summary

Roadmap

More kernels (matmul, convolution, attention).

More backends (GPU runtimes).

Optimizations (not just correctness).
