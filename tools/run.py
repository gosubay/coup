#!/usr/bin/env python3
"""Drive the Coup solve in stages, smallest tree first. Windows, macOS, Linux.

    python tools/run.py             run the plan (Ctrl+C is safe, it resumes)
    python tools/run.py --status    where each stage is, and whether it flattened
    python tools/run.py --bench     just measure this machine and project

Every stage checkpoints and resumes, so stopping and restarting loses nothing.
The iteration counts are a ceiling, not a target: a stage is done when its
NashConv stops falling, which is what --status reports.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOLVE = os.path.join(HERE, "solve_coup.py")

#         name      peek claim       iters
STAGES = (("base",     0,    0,   20_000_000),
          ("claims",   0,    1,   60_000_000),
          ("full",     1,    1,  150_000_000))


def total_ram_gb():
    """Physical RAM, without requiring psutil."""
    try:                                        # Linux, and WSL
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 1048576
    except OSError:
        pass
    try:                                        # Windows
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullExtendedVirtual", ctypes.c_ulonglong)]
        st = S(); st.dwLength = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 1073741824
    except Exception:
        pass
    try:                                        # macOS
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1073741824
    except Exception:
        return 0.0


def run(args, log=None):
    """Run solve_coup.py, echoing output to the screen and to a log file."""
    cmd = [sys.executable, SOLVE] + [str(a) for a in args]
    if log is None:
        return subprocess.call(cmd, cwd=ROOT)
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        return p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "solves"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--only", default="", help="run just this stage, e.g. base")
    ap.add_argument("--iters", type=int, default=0,
                    help="override the iteration ceiling, e.g. --iters 200000 "
                         "for a few-minute trial run")
    ap.add_argument("--checkpoint-every", type=int, default=2000000)
    ap.add_argument("--max-gb", type=float, default=0.0,
                    help="memory ceiling; default is two thirds of RAM")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.bench:
        return run(["bench"])

    if a.status:
        for name, _p, _c, _i in STAGES:
            pkl = os.path.join(a.out, name + ".pkl")
            if not os.path.exists(pkl):
                print(f"{name:<8} not started")
                continue
            print(f"--- {name}")
            run(["stats", "--solve", pkl])
            print()
        return 0

    ram = total_ram_gb()
    max_gb = a.max_gb or (round(ram * 0.66, 1) if ram else 0.0)
    print(f"python  : {sys.version.split()[0]} ({sys.implementation.name})")
    if ram:
        print(f"memory  : {ram:.0f} GB, budget {max_gb:.1f} GB")
    print(f"output  : {a.out}")
    print("Ctrl+C is safe -- every stage checkpoints and resumes.\n")

    if not a.skip_verify:
        print("=== verifying the solver on Kuhn poker (known answer) ===")
        if run(["verify"], os.path.join(a.out, "verify.log")) != 0:
            print("verify failed; not starting the solve")
            return 1
        print()

    for name, peek, claim, iters in STAGES:
        if a.only and a.only != name:
            continue
        if a.iters:
            iters = a.iters
        pkl = os.path.join(a.out, name + ".pkl")
        log = os.path.join(a.out, name + ".log")
        print(f"=== stage {name} (peek={peek} claim={claim}, up to {iters:,}) ===")
        cmd = ["solve", "--peek-memory", peek, "--claim-memory", claim,
               "--iters", iters, "--out", pkl, "--resume",
               "--report-every", min(500000, max(iters // 4, 1)),
               "--checkpoint-every", min(a.checkpoint_every, max(iters, 1)),
               "--exploit-every", 5000000]
        if max_gb:
            cmd += ["--max-gb", max_gb]
        try:
            if run(cmd, log) != 0:
                print(f"stage {name} failed; stopping")
                return 1
        except KeyboardInterrupt:
            print(f"\ninterrupted. {pkl} holds the last checkpoint; "
                  f"re-run this command to resume.")
            return 130
        # opening-action table only: a full dump of the big stages is tens of GB
        run(["export", "--solve", pkl, "--phase", "action", "--min-prob", 0.005,
             "--csv", os.path.join(a.out, name + "-actions.csv")], log)
        print(f"=== stage {name} done ===\n")
    print("ALL STAGES DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
