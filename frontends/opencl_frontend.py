#!/usr/bin/env python3
# OpenCL C host API -> AXIR (regex POC with source-order + real kernel name)
import re, sys, json, argparse, pathlib

def opencl_to_axir(src: str):
    ops_pos = []  # (position_in_source, op_dict)

    # Hint de départ
    ops_pos.append((0, {"op":"DeviceSelect","device":"auto"}))

    # 1) mémoriser la map: variable kernel -> nom réel ("vector_add")
    kernel_names = {}  # kvar -> "vector_add"
    for m in re.finditer(r'(\w+)\s*=\s*clCreateKernel\([^,]+,\s*"([^"]+)"', src):
        kvar, name = m.group(1).strip(), m.group(2).strip()
        kernel_names[kvar] = name
        ops_pos.append((m.start(), {"op":"Comment", "text": f"kernel {kvar} -> {name}"}))

    # 2) buffers (malloc)
    for m in re.finditer(r'(\w+)\s*=\s*clCreateBuffer\([^,]+,\s*[^,]+,\s*([^,]+)\s*,\s*[^,]+,\s*[^)]+\)', src):
        var, bytes_expr = m.group(1).strip(), m.group(2).strip()
        ops_pos.append((m.start(), {"op":"DeviceMalloc","dst":"&"+var,"bytes":bytes_expr}))

    # 3) H2D
    for m in re.finditer(r'clEnqueueWriteBuffer\([^,]+,\s*([^,]+),\s*[^,]+,\s*[^,]+,\s*([^,]+)\s*,\s*([^,)\s]+)', src):
        dst, bytes_expr, src_name = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        ops_pos.append((m.start(), {"op":"Memcpy","dst":dst,"src":src_name,"bytes":bytes_expr,"kind":"H2D"}))

    # 4) set args
    kernel_args = {}  # kvar -> {idx: name}
    for m in re.finditer(r'clSetKernelArg\(\s*(\w+)\s*,\s*(\d+)\s*,\s*[^,]+,\s*&?(\w+)\s*\)', src):
        kvar, idx, arg = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
        d = kernel_args.setdefault(kvar, {})
        d[idx] = arg
        ops_pos.append((m.start(), {"op":"Comment", "text": f"setarg {kvar}[{idx}]={arg}"}))

    # 5) NDRange -> KernelLaunch (avec le **vrai** nom du kernel)
    for m in re.finditer(r'clEnqueueNDRangeKernel\([^,]+,\s*(\w+)\s*,\s*(\d+)\s*,\s*[^,]+,\s*([^,]+)\s*,\s*([^,]+)', src):
        kvar, dim, gsz, lsz = m.group(1).strip(), int(m.group(2)), m.group(3).strip(), m.group(4).strip()
        kname = kernel_names.get(kvar, kvar)  # <-- clé : utiliser "vector_add" si connu
        arg_map = kernel_args.get(kvar, {})
        args = [arg_map[i] for i in sorted(arg_map.keys())] if arg_map else []
        grid  = [gsz, "1", "1"]
        block = [lsz, "1", "1"]
        ops_pos.append((m.start(), {"op":"KernelLaunch","kernel":kname,"grid":grid,"block":block,"args":args}))

    # 6) D2H
    for m in re.finditer(r'clEnqueueReadBuffer\([^,]+,\s*([^,]+),\s*[^,]+,\s*[^,]+,\s*([^,]+)\s*,\s*([^,)\s]+)', src):
        src_name, bytes_expr, dst = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        ops_pos.append((m.start(), {"op":"Memcpy","dst":dst,"src":src_name,"bytes":bytes_expr,"kind":"D2H"}))

    # 7) sync
    for m in re.finditer(r'\bclFinish\(', src):
        ops_pos.append((m.start(), {"op":"DeviceSynchronize"}))

    # 8) free
    for m in re.finditer(r'clReleaseMemObject\(([^)]+)\)', src):
        ptr = m.group(1).strip()
        ops_pos.append((m.start(), {"op":"DeviceFree","ptr":ptr}))

    # ordre du source
    ops_pos.sort(key=lambda x: x[0])
    ops = [op for _, op in ops_pos]
    return {"version":"0.2","meta":{"source_lang":"OpenCL"},"ops":ops}

def main():
    ap = argparse.ArgumentParser(description="OpenCL -> AXIR (POC with ordering + real kernel name)")
    ap.add_argument("input"); ap.add_argument("-o","--output", default=None)
    args = ap.parse_args()
    p = pathlib.Path(args.input)
    if not p.exists(): sys.exit("input not found")
    axir = opencl_to_axir(p.read_text(encoding="utf-8"))
    out = pathlib.Path(args.output) if args.output else p.with_suffix(".axir.json")
    out.write_text(json.dumps(axir, indent=2))
    print(f"[OK] AXIR written: {out}")
if __name__=="__main__":
    main()
