#!/usr/bin/env python3
# AXIR -> CPU executor (NumPy) for demo (robust POC)
import json, argparse, pathlib, sys, re
import numpy as np

def eval_bytes(expr, scalars):
    """very small evaluator for patterns like N*4 or 4096"""
    s = str(expr).replace(" ", "")
    for k, v in scalars.items():
        s = s.replace(k, str(v))
    try:
        if "sizeof(float)" in s: s = s.replace("sizeof(float)", "4")
        if "sizeof(int)"   in s: s = s.replace("sizeof(int)", "4")
        return int(eval(s, {"__builtins__": {}}))
    except Exception:
        return None

def guess_N(ax):
    """try to guess a reasonable vector length N from AXIR"""
    # 0) explicit scalar N
    try:
        return int(ax["types"]["scalars"]["N"]["value"])
    except Exception:
        pass

    max_n = 0
    saw_symbolic_N = False

    # 1) look at KernelLaunch numeric grid/block (prefer larger)
    for op in ax.get("ops", []):
        if op.get("op") == "KernelLaunch":
            for key in ("grid", "block"):
                v = op.get(key, ["", "", ""])[0]
                if isinstance(v, int):
                    max_n = max(max_n, int(v))
                elif isinstance(v, str):
                    sv = v.strip()
                    if re.fullmatch(r"\d+", sv):
                        max_n = max(max_n, int(sv))
                    elif re.search(r"\bN\b", sv):
                        saw_symbolic_N = True

    # 2) look at DeviceMalloc / Memcpy bytes
    for op in ax.get("ops", []):
        if op.get("op") in ("DeviceMalloc", "Memcpy"):
            expr = str(op.get("bytes", ""))
            b = eval_bytes(expr, {})
            if b and b % 4 == 0:
                max_n = max(max_n, b // 4)
            else:
                if re.search(r"\bN\b", expr):
                    saw_symbolic_N = True

    if max_n > 0:
        return max_n
    if saw_symbolic_N:
        return 16  # sensible demo default
    return 16

def run(ax, dump=None):
    scalars = {}
    if "types" in ax and "scalars" in ax["types"]:
        for name, meta in ax["types"]["scalars"].items():
            if "value" in meta:
                try:
                    scalars[name] = int(meta["value"])
                except Exception:
                    pass

    default_N = guess_N(ax)
    # never let N collapse to 1 in demo unless explicit
    if not isinstance(default_N, int) or default_N < 2:
        default_N = 16

    host = {}   # name -> np.array
    device = {} # name -> np.array

    def ensure_host(name, n_elems, dtype="f32"):
        if name in host:
            return
        dt = np.float32 if dtype == "f32" else np.int32
        # deterministic demo data:
        lname = name.lower()
        if lname.endswith("a"):
            host[name] = np.arange(n_elems, dtype=dt)
        elif lname.endswith("b"):
            host[name] = 2 * np.arange(n_elems, dtype=dt)
        else:
            host[name] = np.zeros(n_elems, dtype=dt)

    def dtype_of(buf):
        return (ax.get("types", {}).get("buffers", {}).get(buf, {}) or {}).get("dtype", "f32")

    # First pass: execute ops
    for op in ax["ops"]:
        t = op["op"]

        if t == "DeviceMalloc":
            b = eval_bytes(op.get("bytes", ""), scalars)
            n = b // 4 if (b and b % 4 == 0) else default_N
            name = op["dst"].lstrip("&*")
            device[name] = np.zeros(n, dtype=np.float32)

        elif t == "Memcpy":
            b = eval_bytes(op.get("bytes", ""), scalars)
            n = b // 4 if (b and b % 4 == 0) else None
            kind = op.get("kind")
            if kind == "H2D":
                src = op["src"].lstrip("&*")
                dst = op["dst"].lstrip("&*")
                if n is None:
                    # try to infer from existing device or host
                    if dst in device:
                        n = len(device[dst])
                    elif src in host:
                        n = len(host[src])
                    else:
                        n = default_N
                ensure_host(src, n, dtype_of(src))
                device[dst] = host[src].copy()
            elif kind == "D2H":
                src = op["src"].lstrip("&*")
                dst = op["dst"].lstrip("&*")
                arr = device.get(src)
                if arr is None:
                    # synthesize if needed (robust POC)
                    device[src] = np.zeros(default_N, dtype=np.float32)
                    arr = device[src]
                host[dst] = arr.copy()
            else:
                # D2D or unknown -> ignore for POC
                pass

        elif t == "KernelLaunch":
            k = op.get("kernel", "")
            args = [a.lstrip("&*") for a in op.get("args", [])]

            # ====== vector_add ======
            if k.lower().startswith("vector_add") or k.lower() == "vector_add":
                # Try to locate inputs by common names or by args
                dA = device.get("dA")
                dB = device.get("dB")
                dCname = None
                # detect output name from args (one ending with 'c')
                for a in args:
                    if a.lower().endswith("c"):
                        dCname = a
                if dCname is None:
                    dCname = "dC"

                # fallback on args for inputs if None
                if dA is None and args:
                    dA = device.get(args[0])
                if dB is None and len(args) > 1:
                    dB = device.get(args[1])

                # synthesize if missing
                if dA is None:
                    n = 0
                    if "hA" in host:
                        n = len(host["hA"])
                    elif "hB" in host:
                        n = len(host["hB"])
                    if n == 0:
                        n = default_N
                    device["dA"] = np.arange(n, dtype=np.float32)
                    dA = device["dA"]
                if dB is None:
                    n = len(dA) if dA is not None else default_N
                    device["dB"] = 2 * np.arange(n, dtype=np.float32)
                    dB = device["dB"]

                # ensure output buffer
                nC = min(len(dA), len(dB)) if (dA is not None and dB is not None) else default_N
                if dCname not in device:
                    device[dCname] = np.zeros(nC, dtype=np.float32)

                device[dCname] = dA + dB

            # ====== saxpy: C = alpha*A + B ======
            elif k.lower().startswith("saxpy") or k.lower() == "saxpy":
                # Expect (tolerant): dA, dB, dC, alpha, N
                dA = device.get("dA")
                dB = device.get("dB")
                dCname = "dC"
                alpha = 2.0  # default
                N = scalars.get("N")

                # Try to detect names and literals from args
                for a in args:
                    la = a.lower()
                    if la.endswith("c"):
                        dCname = a
                    # try to read alpha as a literal
                    try:
                        if a.replace('.', '', 1).isdigit():
                            val = float(a)
                            # simple sanity window to avoid mis-parsing pointers as alpha
                            if 0.0001 <= val <= 1e6:
                                alpha = val
                    except Exception:
                        pass

                # fallback on args for inputs if missing
                if dA is None and len(args) > 0:
                    dA = device.get(args[0])
                if dB is None and len(args) > 1:
                    dB = device.get(args[1])

                # synthesize inputs if still missing
                if dA is None:
                    n = N if isinstance(N, int) else (len(dB) if dB is not None else default_N)
                    device["dA"] = np.arange(n, dtype=np.float32)
                    dA = device["dA"]
                if dB is None:
                    n = len(dA) if dA is not None else (N if isinstance(N, int) else default_N)
                    device["dB"] = 2 * np.arange(n, dtype=np.float32)
                    dB = device["dB"]

                nC = min(len(dA), len(dB))
                device[dCname] = alpha * dA + dB

            # ====== reduce_sum: Out = sum(A) ======
            elif k.lower().startswith("reduce_sum") or k.lower() == "reduce_sum":
                # Expect (tolerant): dA, dOut, N
                dA = device.get("dA")
                out_name = "dOut"

                # try to pick names from args
                if dA is None and len(args) > 0:
                    dA = device.get(args[0])
                for a in args:
                    if a.lower().endswith("out"):
                        out_name = a

                # synthesize input if missing
                if dA is None:
                    n = default_N
                    device["dA"] = np.arange(n, dtype=np.float32)
                    dA = device["dA"]

                # perform reduction on CPU
                s = float(np.sum(dA.astype(np.float32)))
                device[out_name] = np.array([s], dtype=np.float32)

            else:
                # other kernels: no-op for POC
                pass

        elif t == "DeviceFree":
            name = op["ptr"].lstrip("&*")
            device.pop(name, None)

    if dump:
        arr = host.get(dump) or device.get(dump)
        if arr is None:
            raise SystemExit(f"dump target not found: {dump}")
        return arr

    return {"host": host, "device": device, "scalars": scalars}

def main():
    ap = argparse.ArgumentParser(description="AXIR -> CPU (NumPy) executor (robust POC)")
    ap.add_argument("axir")
    ap.add_argument("--dump", help="buffer name to dump (e.g., hC)")
    ap.add_argument("--out", help="path to save .npy")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    ax = json.loads(pathlib.Path(a.axir).read_text(encoding="utf-8"))

    if a.summary:
        res = run(ax)
        h = {k: list(v[:5]) for k, v in res["host"].items()}
        d = {k: list(v[:5]) for k, v in res["device"].items()}
        print("[SUMMARY] host(head):", h)
        print("[SUMMARY] device(head):", d)
        return

    if a.dump:
        arr = run(ax, dump=a.dump)
        if a.out:
            np.save(a.out, arr)
            print(f"[OK] saved {a.dump} -> {a.out} shape={arr.shape}")
        else:
            print(arr)
    else:
        run(ax)
        print("[OK] executed AXIR on CPU (no dump)")

if __name__ == "__main__":
    main()
