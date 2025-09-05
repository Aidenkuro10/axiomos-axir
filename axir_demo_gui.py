#!/usr/bin/env python3
# Simple one-button GUI to run the full AXIR demo
import os, sys, subprocess, threading, queue, pathlib, shlex
import tkinter as tk
from tkinter import ttk

REPO = pathlib.Path(__file__).resolve().parent

# (cmd, workdir) tuples; we will skip steps whose input files are missing.
STEPS = [
    # --- CUDA ---
    (r'py -3 frontends\cuda_frontend.py demos\vector_add\vector_add.cu -o demos\vector_add\vadd_from_cuda.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\vector_add\vadd_from_cuda.axir.json --summary', REPO),

    (r'py -3 frontends\cuda_frontend.py demos\saxpy\saxpy.cu -o demos\saxpy\saxpy_from_cuda.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\saxpy\saxpy_from_cuda.axir.json --summary', REPO),

    (r'py -3 frontends\cuda_frontend.py demos\reduce\reduce_sum.cu -o demos\reduce\reduce_from_cuda.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\reduce\reduce_from_cuda.axir.json --summary', REPO),

    # --- HIP (AMD) ---
    (r'py -3 frontends\hip_frontend.py demos\vector_add\vector_add.hip.cpp -o demos\vector_add\vadd_from_hip.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\vector_add\vadd_from_hip.axir.json --summary', REPO),

    (r'py -3 frontends\hip_frontend.py demos\saxpy\saxpy.hip.cpp -o demos\saxpy\saxpy_from_hip.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\saxpy\saxpy_from_hip.axir.json --summary', REPO),

    (r'py -3 frontends\hip_frontend.py demos\reduce\reduce_sum.hip.cpp -o demos\reduce\reduce_from_hip.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\reduce\reduce_from_hip.axir.json --summary', REPO),

    # --- OpenCL ---
    (r'py -3 frontends\opencl_frontend.py demos\vector_add\vector_add.cl -o demos\vector_add\vadd_from_opencl.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\vector_add\vadd_from_opencl.axir.json --summary', REPO),

    (r'py -3 frontends\opencl_frontend.py demos\saxpy\saxpy.cl -o demos\saxpy\saxpy_from_opencl.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\saxpy\saxpy_from_opencl.axir.json --summary', REPO),

    (r'py -3 frontends\opencl_frontend.py demos\reduce\reduce_sum.cl -o demos\reduce\reduce_from_opencl.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\reduce\reduce_from_opencl.axir.json --summary', REPO),

    # --- SYCL (Intel/oneAPI) ---
    (r'py -3 frontends\sycl_frontend.py demos\vector_add\vector_add.sycl.cpp -o demos\vector_add\vadd_from_sycl.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\vector_add\vadd_from_sycl.axir.json --summary', REPO),

    (r'py -3 frontends\sycl_frontend.py demos\saxpy\saxpy.sycl.cpp -o demos\saxpy\saxpy_from_sycl.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\saxpy\saxpy_from_sycl.axir.json --summary', REPO),

    (r'py -3 frontends\sycl_frontend.py demos\reduce\reduce_sum.sycl.cpp -o demos\reduce\reduce_from_sycl.axir.json', REPO),
    (r'py -3 backends\cpu_numpy_backend.py demos\reduce\reduce_from_sycl.axir.json --summary', REPO),
]

# Inputs required for each step (so we can skip gracefully if missing)
REQUIRED_INPUTS = {
    'vector_add.cu': 'demos\\vector_add\\vector_add.cu',
    'saxpy.cu': 'demos\\saxpy\\saxpy.cu',
    'reduce_sum.cu': 'demos\\reduce\\reduce_sum.cu',

    'vector_add.hip.cpp': 'demos\\vector_add\\vector_add.hip.cpp',
    'saxpy.hip.cpp': 'demos\\saxpy\\saxpy.hip.cpp',
    'reduce_sum.hip.cpp': 'demos\\reduce\\reduce_sum.hip.cpp',

    'vector_add.cl': 'demos\\vector_add\\vector_add.cl',
    'saxpy.cl': 'demos\\saxpy\\saxpy.cl',
    'reduce_sum.cl': 'demos\\reduce\\reduce_sum.cl',

    'vector_add.sycl.cpp': 'demos\\vector_add\\vector_add.sycl.cpp',
    'saxpy.sycl.cpp': 'demos\\saxpy\\saxpy.sycl.cpp',
    'reduce_sum.sycl.cpp': 'demos\\reduce\\reduce_sum.sycl.cpp',
}

def inputs_present_for(cmd: str) -> bool:
    # naive check: if command references a known input file, ensure it exists
    for k, rel in REQUIRED_INPUTS.items():
        if k in cmd:
            return (REPO / rel).exists()
    return True  # steps that only consume AXIR should pass

def run_cmd_stream(cmd: str, cwd: pathlib.Path, outq: queue.Queue):
    # use shell=True for PowerShell-style "py -3" convenience on Windows
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in iter(proc.stdout.readline, ''):
            outq.put(line.rstrip("\n"))
        proc.wait()
        outq.put(f"[exit {proc.returncode}] {cmd}")
    except Exception as e:
        outq.put(f"[ERROR] {cmd}\n{e}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AXIR Demo — One Button")
        self.geometry("980x600")

        self.run_btn = ttk.Button(self, text="▶ Run Full AXIR Demo", command=self.on_run, width=30)
        self.run_btn.pack(pady=10)

        self.clear_btn = ttk.Button(self, text="Clear Log", command=self.on_clear)
        self.clear_btn.pack(pady=0)

        self.log = tk.Text(self, wrap="word", height=30)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        self.log.insert("end", "This will run CUDA / HIP / OpenCL / SYCL → AXIR → CPU (if the source files exist).\n")
        self.log.insert("end", "Expected host outputs:\n - vector_add: hC = [0,3,6,9,12]\n - saxpy: hC = [0,4,8,12,16]\n - reduce_sum: hOut = 120.0\n\n")

        self.outq = queue.Queue()
        self.worker = None

    def on_clear(self):
        self.log.delete("1.0", "end")

    def on_run(self):
        if self.worker and self.worker.is_alive():
            return
        self.on_clear()
        self.log.insert("end", "=== AXIR Demo starting ===\n\n")
        self.run_btn.config(state="disabled")
        self.worker = threading.Thread(target=self.run_all, daemon=True)
        self.worker.start()
        self.after(100, self.drain_queue)

    def drain_queue(self):
        try:
            while True:
                line = self.outq.get_nowait()
                self.log.insert("end", line + "\n")
                self.log.see("end")
        except queue.Empty:
            pass
        if self.worker and self.worker.is_alive():
            self.after(100, self.drain_queue)
        else:
            self.run_btn.config(state="normal")
            self.log.insert("end", "\n=== Done ===\n")

    def run_all(self):
        # Python version sanity
        self.outq.put("[info] Python: " + sys.version.replace("\n", " "))
        for cmd, cwd in STEPS:
            if not inputs_present_for(cmd):
                self.outq.put(f"[skip] missing input for: {cmd}")
                continue
            self.outq.put(f">>> {cmd}")
            run_cmd_stream(cmd, cwd, self.outq)

if __name__ == "__main__":
    App().mainloop()
