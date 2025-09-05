#!/usr/bin/env python3
# SYCL -> AXIR (regex POC with source-order preservation)
import re, sys, json, argparse, pathlib

def sycl_to_axir(src: str):
    ops_pos = []  # (pos, op)

    # Always start with a device hint (we don't pick a vendor here)
    ops_pos.append((0, {"op":"DeviceSelect","device":"auto"}))

    # --- Detect USM malloc_device with an LHS pointer ---
    # e.g.: float* dA = sycl::malloc_device<float>(N, q);
    MALLOC = re.compile(
        r'(?P<lhs_type>[A-Za-z_]\w*\s*\*\s*)?(?P<lhs>[A-Za-z_]\w*)\s*=\s*sycl::malloc_device\s*<\s*([A-Za-z_]\w*)\s*>\s*\(\s*(?P<count>[^,]+)\s*,\s*[^)]+\)'
    )
    for m in MALLOC.finditer(src):
        lhs = m.group("lhs").strip()
        count = m.group("count").strip()
        # We assume sizeof(float) (POC). You can refine by reading the template type.
        bytes_expr = f"{count}*sizeof(float)"
        ops_pos.append((m.start(), {"op":"DeviceMalloc","dst":f"&{lhs}","bytes":bytes_expr}))

    # --- memcpy via queue.memcpy(dst, src, bytes) ---
    # q.memcpy(hC, dC, N*sizeof(float));
    MEMCPY = re.compile(r'\bmemcpy\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)')
    for m in MEMCPY.finditer(src):
        dst = m.group(1).strip()
        srcv = m.group(2).strip()
        bytes_expr = m.group(3).strip()
        # Infer direction by simple naming convention (h* vs d*)
        def kind_of(dst, src):
            dl, sl = dst.strip().lower(), src.strip().lower()
            if dl.startswith('d') and sl.startswith('h'): return "H2D"
            if dl.startswith('h') and sl.startswith('d'): return "D2H"
            return "D2D"
        kind = kind_of(dst, srcv)
        ops_pos.append((m.start(), {"op":"Memcpy","dst":dst,"src":srcv,"bytes":bytes_expr,"kind":kind}))

    # --- q.wait() -> DeviceSynchronize (optional but nice) ---
    for m in re.finditer(r'\bq\s*\.\s*wait\s*\(\s*\)\s*;', src):
        ops_pos.append((m.start(), {"op":"DeviceSynchronize"}))

    # --- sycl::free(ptr, q); ---
    FREE = re.compile(r'sycl::free\s*\(\s*([^)]+?)\s*,\s*[^)]+\)')
    for m in FREE.finditer(src):
        ptr = m.group(1).strip()
        ops_pos.append((m.start(), {"op":"DeviceFree","ptr":ptr}))

    # --- Kernel launch marker via comment (simple, explicit, robust) ---
    # Pattern in source:  // KERNEL: vector_add(dA,dB,dC,N)
    KMARK = re.compile(r'//\s*KERNEL\s*:\s*([A-Za-z_]\w*)\s*\(\s*([^)]+)\s*\)')
    for m in KMARK.finditer(src):
        kname = m.group(1).strip()
        args_s = m.group(2).strip()
        args = [a.strip() for a in args_s.split(",")] if args_s else []
        # grid/block: we keep 1D with symbolic N for readability
        grid = ["N","1","1"]
        block = ["N","1","1"]
        ops_pos.append((m.start(), {"op":"KernelLaunch","kernel":kname,"grid":grid,"block":block,"args":args}))

    # Sort by appearance in source
    ops_pos.sort(key=lambda x: x[0])
    ops = [op for _, op in ops_pos]
    return {"version":"0.2","meta":{"source_lang":"SYCL"}, "ops":ops}

def main():
    ap = argparse.ArgumentParser(description="SYCL -> AXIR (POC, ordered)")
    ap.add_argument("input")
    ap.add_argument("-o","--output", default=None)
    args = ap.parse_args()
    p = pathlib.Path(args.input)
    if not p.exists():
        sys.exit("input not found")
    src = p.read_text(encoding="utf-8")
    axir = sycl_to_axir(src)
    out = pathlib.Path(args.output) if args.output else p.with_suffix(".axir.json")
    out.write_text(json.dumps(axir, indent=2))
    print(f"[OK] AXIR written: {out}")

if __name__ == "__main__":
    main()
