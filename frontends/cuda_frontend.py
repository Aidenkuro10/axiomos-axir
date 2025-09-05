#!/usr/bin/env python3
# CUDA -> AXIR (regex POC with source-order preservation)
import re, sys, json, argparse, pathlib

def cuda_to_axir(src: str):
    ops_pos = []  # list of (pos_in_source, op_dict)

    # Always start with a DeviceSelect hint
    ops_pos.append((0, {"op":"DeviceSelect","device":"auto"}))

    # cudaMalloc(&dst, bytes)
    for m in re.finditer(r'cudaMalloc\(([^,]+),\s*([^)]+)\)', src):
        dst = m.group(1).strip()
        bytes_expr = m.group(2).strip()
        ops_pos.append((m.start(), {"op":"DeviceMalloc","dst":dst,"bytes":bytes_expr}))

    # cudaMemcpy(dst, src, bytes, kind)
    for m in re.finditer(
        r'cudaMemcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*(cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|cudaMemcpyDeviceToDevice)\)',
        src
    ):
        dst = m.group(1).strip()
        srcv = m.group(2).strip()
        bytes_expr = m.group(3).strip()
        kind_raw = m.group(4)
        kind = "H2D" if "HostToDevice" in kind_raw else ("D2H" if "DeviceToHost" in kind_raw else "D2D")
        ops_pos.append((m.start(), {"op":"Memcpy","dst":dst,"src":srcv,"bytes":bytes_expr,"kind":kind}))

    # cudaDeviceSynchronize()
    for m in re.finditer(r'\bcudaDeviceSynchronize\s*\(\s*\)\s*;', src):
        ops_pos.append((m.start(), {"op":"DeviceSynchronize"}))

    # cudaFree(ptr)
    for m in re.finditer(r'cudaFree\(([^)]+)\)', src):
        ptr = m.group(1).strip()
        ops_pos.append((m.start(), {"op":"DeviceFree","ptr":ptr}))

    # kernel launch: foo<<<grid, block[, sharedMem][, stream]>>>(args)
    LAUNCH_RE = re.compile(
        r'([A-Za-z_]\w*)\s*<<<\s*'
        r'([^,>]+)\s*,\s*'
        r'([^,>]+)'
        r'(?:\s*,\s*[^,>]+)?'
        r'(?:\s*,\s*[^>]+)?'
        r'\s*>>>\s*\(\s*'
        r'([^)]*)\)'
    )
    for m in LAUNCH_RE.finditer(src):
        kernel = m.group(1).strip()
        g1 = m.group(2).strip()
        b1 = m.group(3).strip()
        args_s = m.group(4).strip()
        args = [a.strip() for a in args_s.split(",")] if args_s else []
        ops_pos.append((m.start(), {
            "op":"KernelLaunch",
            "kernel":kernel,
            "grid":[g1,"1","1"],
            "block":[b1,"1","1"],
            "args":args
        }))

    # sort by position in source and extract ops
    ops_pos.sort(key=lambda x: x[0])
    ops = [op for _, op in ops_pos]

    return {"version":"0.2","meta":{"source_lang":"CUDA"}, "ops":ops}

def main():
    ap = argparse.ArgumentParser(description="CUDA -> AXIR (POC, ordered)")
    ap.add_argument("input")
    ap.add_argument("-o","--output", default=None)
    args = ap.parse_args()
    p = pathlib.Path(args.input)
    if not p.exists(): sys.exit("input not found")
    src = p.read_text(encoding="utf-8")
    axir = cuda_to_axir(src)
    out = pathlib.Path(args.output) if args.output else p.with_suffix(".axir.json")
    out.write_text(json.dumps(axir, indent=2))
    print(f"[OK] AXIR written: {out}")

if __name__ == "__main__":
    main()
