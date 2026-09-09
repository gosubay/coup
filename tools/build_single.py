#!/usr/bin/env python3
"""Bundle the package into one standalone coup_solver.py.

A single file with no imports and no repository is easier to run than a clone,
especially on Windows. Everything here is standard library, so the bundle needs
no installation and runs unchanged on PyPy.

    python3 tools/build_single.py [output path]
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

HEAD = '''#!/usr/bin/env python3
"""coup_solver.py -- heads-up Coup solver, everything in one file.

No installation, no dependencies, no repository. Standard library only, so it
runs on any Python 3.8+ and unchanged on PyPy, which is much faster for this.

    python coup_solver.py verify      prove the solver on a game with a known answer
    python coup_solver.py bench       measure your machine, project the run
    python coup_solver.py plan        run the staged plan end to end
    python coup_solver.py strength --solve solves/base.pkl
    python coup_solver.py query    --solve solves/base.pkl --my-coins 7 --hand duke,assassin
    python coup_solver.py export   --solve solves/base.pkl --csv policy.csv --phase action

KNOWN DEFECT -- read this before spending a night on it. The solver plateaus.
Its policy beats a uniform-random opponent about 61% of the time while a plain
Monte-Carlo control agent reaches 90%, and going from 10 million to 20 million
iterations does not move that number at all. `strength` measures it. Until that
climbs, a longer run only computes a weak strategy more precisely.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from array import array
from itertools import combinations

'''

TAIL = '''

# ---------------------------------------------------------------------------
# The package used module prefixes (X.nashconv, E.parse_cards); in a single
# file those resolve to this module itself, so the command bodies above need no
# rewriting.
X = sys.modules[__name__]
E = sys.modules[__name__]


def cmd_strength(a):
    """Win rate against a uniform-random opponent.

    The honest health check, and the one that caught the current defect. It
    cannot fool itself the way an exploitability estimate can: the opponent is
    fixed, trivially defined, and never adapts.
    """
    s = Solver.load(a.solve)
    gf = factory_from(s.tag)
    pol = s.average_strategy()
    rng = random.Random(55)
    w = 0.0
    for _ in range(a.games):
        u = play_once(gf, [pol, {}], rng)
        w += 1.0 if u > 0 else (0.0 if u < 0 else 0.5)
    print(f"{a.solve}: t={s.t:,}, {len(s.nodes):,} infosets")
    print(f"\\nwin rate vs uniform random   {w / a.games:6.1%}\\n")
    print("  50%   no better than random")
    print("  61%   this solver at 20M iterations -- the known defect")
    print("  90%   a plain Monte-Carlo control agent, for scale")


def cmd_plan(a):
    """The staged plan, smallest tree first, in this one process."""
    stages = (("base", 0, 0, 20_000_000),
              ("claims", 0, 1, 60_000_000),
              ("full", 1, 1, 150_000_000))
    os.makedirs(a.out, exist_ok=True)
    for name, peek, claim, iters in stages:
        if a.only and a.only != name:
            continue
        if a.iters:
            iters = a.iters
        pkl = os.path.join(a.out, name + ".pkl")
        print(f"\\n=== stage {name} (peek={peek} claim={claim}, up to {iters:,}) ===")
        tag = {"max_turns": 60, "peek_memory": peek, "claim_memory": claim}
        if a.resume and os.path.exists(pkl):
            s = Solver.load(pkl)
            s.game_factory = factory_from(s.tag)
            print(f"resuming at t={s.t:,}")
        else:
            s = Solver(factory_from(tag), seed=0, tag=tag)
        try:
            s.run(iters, report_every=min(500000, max(iters // 4, 1)),
                  checkpoint=pkl,
                  checkpoint_every=min(2000000, max(iters, 1)),
                  exploit_every=0, max_gb=a.max_gb)
        except KeyboardInterrupt:
            s.save(pkl)
            print(f"\\ninterrupted; {pkl} holds the checkpoint, --resume continues")
            return 130
        print(f"saved {pkl}: t={s.t:,}, {len(s.nodes):,} infosets")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", help="check the solver on Kuhn poker")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("bench", help="measure this machine and project the plan")
    p.add_argument("--iters", type=int, default=20000)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("strength", help="win rate vs uniform random")
    p.add_argument("--solve", required=True)
    p.add_argument("--games", type=int, default=15000)
    p.set_defaults(fn=cmd_strength)

    p = sub.add_parser("plan", help="run the staged plan")
    p.add_argument("--out", default="solves")
    p.add_argument("--only", default="", help="just one stage, e.g. base")
    p.add_argument("--iters", type=int, default=0, help="override the ceiling")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-gb", type=float, default=0.0)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("solve")
    p.add_argument("--iters", type=int, default=1000000)
    p.add_argument("--out", default="coup_solve.pkl")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--peek-memory", type=int, default=1)
    p.add_argument("--claim-memory", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-every", type=int, default=50000)
    p.add_argument("--checkpoint-every", type=int, default=250000)
    p.add_argument("--exploit-every", type=int, default=0)
    p.add_argument("--exploit-iters", type=int, default=30000)
    p.add_argument("--exploit-games", type=int, default=20000)
    p.add_argument("--max-gb", type=float, default=0.0)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("exploit")
    p.add_argument("--solve", required=True)
    p.add_argument("--br-iters", type=int, default=900000)
    p.add_argument("--games", type=int, default=30000)
    p.set_defaults(fn=cmd_exploit)

    p = sub.add_parser("stats")
    p.add_argument("--solve", required=True)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("query")
    p.add_argument("--solve", required=True)
    p.add_argument("--my-lives", type=int, default=2)
    p.add_argument("--opp-lives", type=int, default=2)
    p.add_argument("--my-coins", type=int, default=2)
    p.add_argument("--opp-coins", type=int, default=2)
    p.add_argument("--hand", default="duke,captain")
    p.add_argument("--face-up", default="")
    p.add_argument("--peek", default="")
    p.add_argument("--my-claims", default="")
    p.add_argument("--opp-claims", default="")
    p.add_argument("--phase", default="action", choices=list(PHASES))
    p.add_argument("--vs", default="")
    p.add_argument("--pool", default="")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("export")
    p.add_argument("--solve", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--phase", default="", choices=[""] + list(PHASES))
    p.add_argument("--min-prob", type=float, default=0.0)
    p.set_defaults(fn=cmd_export)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
'''


def strip(src):
    """Drop intra-package imports, including indented multi-line ones."""
    out, skip = [], False
    for line in src.splitlines():
        if skip:
            skip = ")" not in line
            continue
        if re.match(r"\s*from \.\w+ import", line) or re.match(r"\s*from coup[. ]", line):
            skip = "(" in line and ")" not in line
            continue
        if "from __future__" in line or re.match(r"\s*sys\.path\.insert", line):
            continue
        out.append(line)
    return "\n".join(out)


def banner(name):
    return "\n\n# " + "=" * 74 + f"\n# {name}\n# " + "=" * 74 + "\n"


def read(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return strip(f.read())


def build():
    # keep everything from the PHASES table onward: the command bodies plus the
    # constants they close over. Drop main(), which the bundle redefines.
    cli = read("solve_coup.py")
    cli = "PHASES = {" + cli.split("PHASES = {", 1)[1]
    cli = cli.split("def main()")[0].rstrip()
    return "".join([
        HEAD,
        banner("GAME"), read("coup/game.py"),
        banner("KUHN POKER -- the correctness test"), read("coup/kuhn.py"),
        banner("SOLVER"), read("coup/solver.py"),
        banner("EXPLOITABILITY"), read("coup/exploit.py"),
        banner("LOOKUP AND EXPORT"), read("coup/export.py"),
        banner("COMMANDS"), cli,
        TAIL,
    ])


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "coup_solver.py"
    text = build()
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {dest}: {len(text.splitlines()):,} lines, "
          f"{len(text) / 1024:.0f} KB, zero dependencies")
