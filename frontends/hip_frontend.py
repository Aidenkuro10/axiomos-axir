#!/usr/bin/env python3
# HIP -> AXIR (regex POC with source-order preservation)
import re, sys, json, argparse, pathlib

def hip_to_axir(src: str):
    ops_pos = []  # list of (pos_in_source, op_dict)

    # Start with a device hint
    ops_pos.append((0, {"op": "DeviceSelect", "device": "auto"}))

    # hipMalloc(&dst, bytes)
    for m in re.finditer(r'hipMalloc\(([^,]+),\s*([^)]+)\)', src):
        dst = m.group(1).strip()
        bytes_expr = m.group(2).strip()
        ops_pos.append((m.start(), {"op": "DeviceMalloc", "dst": dst, "bytes": bytes_expr}))

    # hipMemcpy(dst, src, bytes, kind)
    for m in re.finditer(
        r'hipMemcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*(hipMemcpyHostToDevice|hipMemcpyDeviceToHost|hipMemcpyDeviceToDevice)\)',
        src
    ):
        dst = m.group(1).strip()
        srcv = m.group(2).strip()
        bytes_expr = m.group(3).strip()
        kind_raw = m.group(4)
        kind = "H2D" if "HostToDevice" in kind_raw else ("D2H" if "DeviceToHost" in kind_raw else "D2D")
        ops_pos.append((m.start(), {"op": "Memcpy", "dst": dst, "src": srcv, "bytes": bytes_expr, "kind": kind}))

    # hipDeviceSynchronize()
    for m in re.finditer(r'\bhipDeviceSynchronize\s*\(\s*\)\s*;', src):
        ops_pos.append((m.start(), {"op": "DeviceSynchronize"}))

    # hipFree(ptr)
    for m in re.finditer(r'hipFree\(([^)]+)\)', src):
        ptr = m.group(1).strip()
        ops_pos.append((m.start(), {"op": "DeviceFree", "ptr": ptr}))

    # hipLaunchKernelGGL(kernel, dim3(gridX[,Y[,Z]]), dim3(blockX[,Y[,Z]]), shared, stream, args...)
    LAUNCH = re.compile(
        r'hipLaunchKernelGGL\s*\(\s*'          # hipLaunchKernelGGL(
        r'([A-Za-z_]\w*)\s*,\s*'               #   kernel
        r'dim3\s*\(\s*([^,)]+)(?:\s*,\s*[^,)]+){0,2}\s*\)\s*,\s*'  # dim3(gridX[,Y[,Z]])
        r'dim3\s*\(\s*([^,)]+)(?:\s*,\s*[^,)]+){0,2}\s*\)\s*,\s*'  # dim3(blockX[,Y[,Z]])
        r'(?:[^,]*,\s*){2}'                    # skip sharedMem, stream
        r'(.*?)\)'                             # args (greedy but stops at ')'
        , re.DOTALL
    )
    for m in LAUNCH.finditer(src):
        kernel = m.group(1).strip()
        g1 = m.group(2).strip()
        b1 = m.group(3).strip()
        args_s = m.group(4).strip()
        # split args safely (flat comma split is enough for our POC inputs)
        args = [a.strip() for a in args_s.split(",")] if args_s else []
        ops_pos.append((m.start(), {
            "op": "KernelLaunch",
            "kernel": kernel,
            "grid": [g1, "1", "1"],
            "block": [b1, "1", "1"],
            "args": args
        }))

    # Sort by source position and extract ops
    ops_pos.sort(key=lambda x: x[0])
    ops = [op for _, op in ops_pos]
    return {"version": "0.2", "meta": {"source_lang": "HIP"}, "ops": ops}

def main():
    parser = argparse.ArgumentParser(description="HIP -> AXIR (POC, ordered)")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    p = pathlib.Path(args.input)
    if not p.exists():
        sys.exit("input not found")

    src = p.read_text(encoding="utf-8")
    axir = hip_to_axir(src)
    out = pathlib.Path(args.output) if args.output else p.with_suffix(".axir.json")
    out.write_text(json.dumps(axir, indent=2))
    print(f"[OK] AXIR written: {out}")

if __name__ == "__main__":
    main()
