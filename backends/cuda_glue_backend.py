#!/usr/bin/env python3
# AXIR -> CUDA glue (POC)
import sys, json, argparse, pathlib

def emit(ax):
    L = ["#include <cuda_runtime.h>", "// AXIR -> CUDA (glue POC)\n"]
    for op in ax.get("ops", []):
        t = op["op"]
        if t == "DeviceSelect":
            L.append(f"// DeviceSelect: {op.get('device','auto')}")
        elif t == "DeviceMalloc":
            L.append(f"cudaMalloc({op['dst']}, {op['bytes']});")
        elif t == "Memcpy":
            k = {"H2D":"cudaMemcpyHostToDevice","D2H":"cudaMemcpyDeviceToHost","D2D":"cudaMemcpyDeviceToDevice"}[op["kind"]]
            L.append(f"cudaMemcpy({op['dst']}, {op['src']}, {op['bytes']}, {k});")
        elif t == "DeviceSynchronize":
            L.append("cudaDeviceSynchronize();")
        elif t == "DeviceFree":
            L.append(f"cudaFree({op['ptr']});")
        elif t == "KernelLaunch":
            g = op["grid"][0]; b = op["block"][0]
            args = ", ".join(op.get("args", []))
            L.append(f"{op['kernel']}<<<{g},{b},0,0>>>({args});")
        elif t == "Comment":
            L.append(f"// {op['text']}")
    L.append("")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("axir"); ap.add_argument("-o","--output", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.axir); ax = json.loads(p.read_text(encoding="utf-8"))
    out = pathlib.Path(a.output) if a.output else p.with_suffix(".cu")
    out.write_text(emit(ax))
    print(f"[OK] CUDA glue written: {out}")

if __name__=="__main__": main()
