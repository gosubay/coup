#!/usr/bin/env python3
"""Solve heads-up Coup with CFR+, and prove the answer is actually solved.

    # correctness check first -- Kuhn poker has a known analytic equilibrium
    python3 tools/solve_coup.py verify

    # the solve. watch NashConv; stop when it flattens, NOT on iteration count
    python3 tools/solve_coup.py solve --iters 5000000 --out coup_solve.pkl

    python3 tools/solve_coup.py solve --resume --iters 5000000 --out coup_solve.pkl
    python3 tools/solve_coup.py exploit --solve coup_solve.pkl
    python3 tools/solve_coup.py stats   --solve coup_solve.pkl
    python3 tools/solve_coup.py query   --solve coup_solve.pkl \
            --my-lives 2 --opp-lives 2 --my-coins 4 --opp-coins 6 --hand duke,captain
    python3 tools/solve_coup.py export  --solve coup_solve.pkl --csv policy.csv
"""

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coup.solver import Solver, policy_stats          # noqa: E402
from coup import exploit as X                         # noqa: E402
from coup import export as E                          # noqa: E402
from coup.game import (new_game, PHASE_ACTION, PHASE_RESPOND,   # noqa: E402
                       PHASE_BLOCK_RESP, PHASE_DISCARD, PHASE_EXCHANGE)

PHASES = {"action": PHASE_ACTION, "respond": PHASE_RESPOND,
          "answer-block": PHASE_BLOCK_RESP, "discard": PHASE_DISCARD,
          "exchange-keep": PHASE_EXCHANGE}


def factory_from(tag):
    return functools.partial(new_game, tag.get("max_turns", 60),
                             tag.get("peek_memory", 1),
                             tag.get("claim_memory", 1))


def cmd_verify(a):
    """Kuhn poker: known game value -1/18, known equilibrium structure."""
    from coup.kuhn import new_kuhn, GAME_VALUE_P0
    s = Solver(new_kuhn, seed=1)
    print(f"{'iters':>9} {'selfplay EV':>13} {'NashConv':>11}")
    for n in (5000, 20000, 75000, 200000):
        s.run(n)
        pol = s.average_strategy()
        ev = X.evaluate(new_kuhn, pol, pol, 60000, seed=2)
        nc = X.nashconv(new_kuhn, pol, br_iters=30000, games=30000)
        print(f"{s.t:>9,} {ev:>+13.5f} {nc:>11.5f}")
    print(f"\ntarget game value to P0 = -1/18 = {GAME_VALUE_P0:+.5f}")
    print("NashConv must be falling toward 0. If it is not, do not trust the "
          "Coup numbers either.")


# Tree sizes measured on this codebase; they are a property of the game, not of
# your hardware, so they carry over. `full` is a range because it was still
# discovering infosets when the measurement stopped.
STAGE_PLAN = (
    ("base",   0, 0,  20_000_000, (380_000, 380_000)),
    ("claims", 0, 1,  60_000_000, (6_000_000, 6_000_000)),
    ("full",   1, 1, 150_000_000, (15_000_000, 30_000_000)),
)
BYTES_PER_INFOSET = 443


def cmd_bench(a):
    """Measure this machine, then project the whole plan onto it."""
    import platform
    import time as _t

    impl = platform.python_implementation()
    print(f"runtime : {impl} {platform.python_version()} on {platform.machine()}")
    if impl != "PyPy":
        print("          (PyPy runs this same code with no changes and is "
              "typically several times faster --")
        print("           install it and re-run this bench to see the real "
              "number for your machine)")
    try:
        with open("/proc/meminfo") as f:
            total_gb = int(f.readline().split()[1]) / 1048576
        print(f"memory  : {total_gb:.0f} GB")
    except OSError:
        total_gb = None
    print()

    # Only speed is measured here. Memory per infoset is a property of the code,
    # not the machine, and measuring it in-process is unreliable because freed
    # pages stay resident between configurations.
    rates = {}
    print(f"{'config':<10} {'it/s':>9}")
    for name, peek, claim, _, _sz in STAGE_PLAN:
        s = Solver(factory_from({"peek_memory": peek, "claim_memory": claim}), seed=1)
        s.run(2000)                                   # warm up, then measure
        t0 = _t.time(); s.run(a.iters); el = _t.time() - t0
        rates[name] = a.iters / el
        print(f"{name:<10} {rates[name]:>9,.0f}")

    print(f"\nprojected for the full plan on this machine:")
    print(f"  {'stage':<8} {'iters':>12} {'wall time':>11} {'peak RAM':>10}")
    total = 0.0
    for name, _p, _c, iters, (lo, hi) in STAGE_PLAN:
        hours = iters / rates[name] / 3600
        total += hours
        gb_lo = lo * BYTES_PER_INFOSET / 1073741824
        gb_hi = hi * BYTES_PER_INFOSET / 1073741824
        ram = f"{gb_lo:.1f} GB" if lo == hi else f"{gb_lo:.0f}-{gb_hi:.0f} GB"
        print(f"  {name:<8} {iters:>12,} {hours:>10.1f} h {ram:>10}")
    print(f"  {'total':<8} {'':>12} {total:>10.1f} h")

    if total_gb is not None:
        worst = STAGE_PLAN[-1][4][1] * BYTES_PER_INFOSET / 1073741824
        if worst > total_gb * 0.66:
            print(f"\n  Note: `full` may not fit in {total_gb:.0f} GB. Pass "
                  f"--max-gb {total_gb*0.66:.1f} and it will checkpoint and stop")
            print("  cleanly there rather than being OOM-killed; `base` and "
                  "`claims` are unaffected.")


def cmd_solve(a):
    tag = {"max_turns": a.max_turns, "peek_memory": a.peek_memory,
           "claim_memory": a.claim_memory}
    if a.resume and os.path.exists(a.out):
        s = Solver.load(a.out)
        s.game_factory = factory_from(s.tag)
        print(f"resuming {a.out} at t={s.t:,} ({len(s.nodes):,} infosets)")
    else:
        s = Solver(factory_from(tag), seed=a.seed, tag=tag)
        print(f"new solve: max_turns={a.max_turns} peek_memory={a.peek_memory} "
              f"claim_memory={a.claim_memory}")
        print("peek_memory=1 keeps the cards you hand back in an Exchange, which "
              "is what makes card-removal reasoning possible.")
        print("claim_memory=1 keeps which roles each side has claimed since their "
              "hand last changed, which is what makes bluff-consistency "
              "reasoning possible.\n")

    gf = s.game_factory
    exploit_fn = None
    if a.exploit_every:
        def exploit_fn(sol):
            # The responder has to be trained in proportion to the policy it is
            # attacking; a fixed small budget measures its own training, not the
            # solve. See nashconv_ladder for why.
            budget = max(a.exploit_iters, 3 * len(sol.nodes))
            return X.nashconv(gf, sol.average_strategy(),
                              br_iters=budget, games=a.exploit_games), budget

    s.run(a.iters, report_every=a.report_every, checkpoint=a.out,
          checkpoint_every=a.checkpoint_every, exploit_every=a.exploit_every,
          exploit_fn=exploit_fn, max_gb=a.max_gb)
    print(f"\nsaved {a.out}: t={s.t:,}, {len(s.nodes):,} infosets")


def cmd_exploit(a):
    s = Solver.load(a.solve); gf = factory_from(s.tag)
    pol = s.average_strategy()
    print(f"{a.solve}: t={s.t:,}, {len(s.nodes):,} infosets\n")
    print(f"self-play EV to P0      {X.self_play_value(gf, pol, a.games):+.4f}\n")
    print("NashConv, raising the best-responder's budget until it stops rising:")
    budgets = tuple(b for b in (30000, 100000, 300000, 900000, 2700000)
                    if b <= a.br_iters) or (a.br_iters,)
    nc, ok, _rows = X.nashconv_ladder(gf, pol, budgets=budgets, games=a.games,
                                      verbose=True)
    if ok:
        print(f"\n  NashConv >= {nc:.4f}   (0 means solved; the responder "
              "plateaued, so this is a fair bound)\n")
    else:
        print(f"\n  NashConv >= {nc:.4f}   -- and STILL RISING at the largest "
              "budget, so the\n  true exploitability is higher than this. Raise "
              "--br-iters.\n")
    print("cheap deviation check:")
    for name, ev, gain in X.quick_deviations(gf, pol, games=a.games):
        print(f"  {name:<34} EV {ev:+.4f}" + ("" if gain == 0 else f"   gain {gain:+.4f}"))


def cmd_stats(a):
    s = Solver.load(a.solve)
    st = policy_stats(s.average_strategy())
    n = st["infosets"]
    print(f"iterations : {s.t:,}")
    print(f"infosets   : {n:,}")
    print(f"  pure  (one right answer)      {st['pure']:>9,}  {st['pure']/n:6.1%}")
    print(f"  near-pure                     {st['near_pure']:>9,}  {st['near_pure']/n:6.1%}")
    print(f"  mixed (answer IS a frequency) {st['mixed']:>9,}  {st['mixed']/n:6.1%}")
    if s.trace:
        print("\nNashConv trace:")
        for t, nc, sec in s.trace:
            print(f"  t={t:>11,}  {nc:+.5f}   ({sec/60:.1f} min)")
    else:
        print("\nno NashConv trace -- you have no evidence this solve is finished.")


def cmd_query(a):
    s = Solver.load(a.solve)
    rows = E.lookup(
        s.average_strategy(), my_lives=a.my_lives, opp_lives=a.opp_lives,
        my_coins=a.my_coins, opp_coins=a.opp_coins,
        my_hand=E.parse_cards(a.hand), revealed=E.parse_cards(a.face_up),
        peek=E.parse_cards(a.peek),
        my_claims=E.claim_mask(a.my_claims), opp_claims=E.claim_mask(a.opp_claims),
        phase=PHASES[a.phase],
        pend=E.ACTION_ID[a.vs.lower()] if a.vs else None,
        pool=E.parse_cards(a.pool) if a.pool else None)
    print(f"you {a.my_lives} inf / {a.my_coins}c ({a.hand})   "
          f"opp {a.opp_lives} inf / {a.opp_coins}c"
          + (f"   face-up: {a.face_up}" if a.face_up else "")
          + (f"   you claimed: {a.my_claims}" if a.my_claims else "")
          + (f"   opp claimed: {a.opp_claims}" if a.opp_claims else ""))
    if rows is None:
        print("\nthat situation never came up in training -- no reliable answer")
        return
    print()
    for label, p in rows:
        print(f"  {label:<24} {p:6.1%}  {'#' * int(round(p * 40))}")


def cmd_export(a):
    s = Solver.load(a.solve)
    n = E.export_csv(s.average_strategy(), a.csv,
                     phase=PHASES[a.phase] if a.phase else None,
                     min_prob=a.min_prob)
    print(f"wrote {n:,} rows to {a.csv}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", help="check the solver on Kuhn poker")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("bench", help="measure this machine and project the plan")
    p.add_argument("--iters", type=int, default=20000)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("solve")
    p.add_argument("--iters", type=int, default=1000000)
    p.add_argument("--out", default="coup_solve.pkl")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--peek-memory", type=int, default=1,
                   help="1 = remember the cards you handed back in an Exchange "
                        "(card removal); 0 = the old solve's information")
    p.add_argument("--claim-memory", type=int, default=1,
                   help="1 = remember which roles each side claimed since their "
                        "hand last changed (bluff consistency); 0 = forget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gb", type=float, default=0.0,
                   help="stop and checkpoint if resident memory passes this, "
                        "instead of being OOM-killed. 0 = no limit")
    p.add_argument("--report-every", type=int, default=50000)
    p.add_argument("--checkpoint-every", type=int, default=250000)
    p.add_argument("--exploit-every", type=int, default=500000)
    p.add_argument("--exploit-iters", type=int, default=30000)
    p.add_argument("--exploit-games", type=int, default=20000)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("exploit")
    p.add_argument("--solve", required=True)
    p.add_argument("--br-iters", type=int, default=900000,
                   help="largest best-responder budget to try")
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
    p.add_argument("--peek", default="", help="cards you handed back in an Exchange")
    p.add_argument("--my-claims", default="",
                   help="roles you have claimed since your hand last changed")
    p.add_argument("--opp-claims", default="",
                   help="roles the opponent has claimed since their hand last changed")
    p.add_argument("--phase", default="action", choices=list(PHASES))
    p.add_argument("--vs", default="", help="the action you are responding to")
    p.add_argument("--pool", default="", help="your 4-card pool for exchange-keep")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("export")
    p.add_argument("--solve", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--phase", default="", choices=[""] + list(PHASES))
    p.add_argument("--min-prob", type=float, default=0.0)
    p.set_defaults(fn=cmd_export)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
