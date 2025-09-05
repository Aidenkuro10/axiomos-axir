#!/usr/bin/env python3
# AXIR -> HIP glue (POC)
import sys, json, argparse, pathlib

def emit(ax):
    L = ["#include <hip/hip_runtime.h>", "// AXIR -> HIP (glue POC)\n"]
    for op in ax.get("ops", []):
        t = op["op"]
        if t == "DeviceSelect":
            L.append(f"// DeviceSelect: {op.get('device','auto')}")
        elif t == "DeviceMalloc":
            L.append(f"hipMalloc({op['dst']}, {op['bytes']});")
        elif t == "Memcpy":
            k = {"H2D":"hipMemcpyHostToDevice","D2H":"hipMemcpyDeviceToHost","D2D":"hipMemcpyDeviceToDevice"}[op["kind"]]
            L.append(f"hipMemcpy({op['dst']}, {op['src']}, {op['bytes']}, {k});")
        elif t == "DeviceSynchronize":
            L.append("hipDeviceSynchronize();")
        elif t == "DeviceFree":
            L.append(f"hipFree({op['ptr']});")
        elif t == "KernelLaunch":
            g = ",".join(op["grid"])
            b = ",".join(op["block"])
            args = ", ".join(op.get("args", []))
            L.append(f"hipLaunchKernelGGL({op['kernel']}, dim3({g}), dim3({b}), 0, 0, {args});")
        elif t == "Comment":
            L.append(f"// {op['text']}")
    L.append("")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("axir"); ap.add_argument("-o","--output", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.axir); ax = json.loads(p.read_text(encoding="utf-8"))
    out = pathlib.Path(a.output) if a.output else p.with_suffix(".hip.cpp")
    out.write_text(emit(ax))
    print(f"[OK] HIP glue written: {out}")

if __name__=="__main__": main()
