# Coup solver

A heads-up Coup solver that reports whether its own answer is finished.

The previous solve did not, and that turned out to be the whole problem: a
five-line greedy heuristic beat it by 17 percentage points of win rate, while
its own stored regrets already said "Coup more" and the average strategy had
not caught up.

## Quick start

```bash
# 1. prove the solver is correct on a game with a known analytic answer
python3 tools/solve_coup.py verify

# 2. solve. watch NashConv. stop when it flattens -- NOT on iteration count
python3 tools/solve_coup.py solve --iters 5000000 --out coup_solve.pkl

# resume any time; checkpoints are written as it goes
python3 tools/solve_coup.py solve --resume --iters 5000000 --out coup_solve.pkl

# 3. check it
python3 tools/solve_coup.py exploit --solve coup_solve.pkl
python3 tools/solve_coup.py stats   --solve coup_solve.pkl

# 4. use it
python3 tools/solve_coup.py query --solve coup_solve.pkl \
    --my-lives 2 --opp-lives 2 --my-coins 7 --opp-coins 3 --hand duke,assassin

python3 tools/solve_coup.py export --solve coup_solve.pkl --csv policy.csv
```

## What the state includes

Every decision is keyed on exactly what the player legitimately knows:

| | |
|---|---|
| your influence / opponent's influence | 1 or 2 each |
| your coins / opponent's coins | 0-12 |
| your hand | the cards you hold |
| face-up cards | everything already revealed |
| **cards you handed back in an Exchange** | **new** -- this is the card-removal information |
| what you are responding to | pending action, pending block |

## What is new versus the old solve

The old key already had influence, coins, hand and face-up cards. Three things
were missing, and they are the reason for a rebuild rather than a longer run:

1. **Which cards to keep after an Exchange.** The old solve never had this as a
   decision -- max hand size across all 219,187 of its infosets was 2, so the
   4-card pool never existed. Measured worth: about **0.069 win units**, enough
   to move Exchange from the worst opening to the second best.
2. **Which influence to reveal when you lose one.** Also automated, and
   calibration against the old file rejected any strategic rule for it.
3. **Peek memory.** The cards you saw during an Exchange and returned to the
   deck. Run with `--peek-memory 0` to reproduce the old information set.

## Why it converges when the old one did not

- **Regret matching+** -- cumulative regrets are clamped at zero, so a bad early
  stretch stops dragging the answer forever.
- **Linear averaging** -- iteration *t* counts with weight *t*, so recent, better
  iterations outvote ancient ones. This is the specific defect that left the old
  solve coupling at 55.7% when its own regrets wanted far more.
- **Outcome sampling** rather than external sampling. External sampling
  enumerates every action of the player being updated, and Coup games are long
  enough that this explodes combinatorially -- measured at over 900 seconds for
  40 traversals before the switch.

## Reading NashConv

NashConv is how much a best response beats the strategy by, summed over both
seats. **Zero means solved.** Anything else is how far you have to go.

It is a *lower bound*: the best response is itself approximated, so the true
distance is larger. On Kuhn poker, where the exact answer is computable by brute
force, this estimator recovers 40-85% of the true value and -- the part that
matters -- falls monotonically alongside it:

| solve iterations | exact | estimated |
|---|---|---|
| 2,000 | 0.178 | 0.117 |
| 20,000 | 0.059 | 0.050 |
| 200,000 | 0.023 | 0.0095 |
| 600,000 | 0.011 | 0.0046 |

So: trust the trend and the order of magnitude, not the last decimal. Monte
Carlo also puts a resolution floor near `2/sqrt(games)` on it. It is built to
answer "is this badly off?" reliably, which is the question that went unasked
last time.

`exploit` also runs a cheap deviation check -- do dumb fixed rules beat the
solve? That is what caught the old one, and it takes seconds.

## Correctness

`verify` solves Kuhn poker, whose value to player 0 is exactly -1/18 and whose
equilibrium structure is known: never bet the queen first, bet the king three
times as often as the jack, always call a bet with the king, always fold the
jack. The solver reproduces all of it. The Coup engine is separately checked for
card conservation -- every one of the 15 cards accounted for at every node --
across random playouts, with all six decision phases reachable.

## Cost, measured rather than guessed

CPython runs 2,400-3,700 iterations/sec on this engine. A real run looks like:

```
  t=  100,000  infosets=  404,416   3747 it/s
  t=  200,000  infosets=  607,798   NashConv=+1.46
  t=  400,000  infosets=  891,623   NashConv=+1.38
  t=  600,000  infosets=1,101,358   NashConv=+1.37
```

Read that honestly: after 600k iterations the solve is **nowhere near done**.
Infosets are still being discovered as fast as they are being trained, and the
full tree is around 4.6M. Expect to need **tens of millions of iterations** --
hours, not minutes. That is the real cost of wanting every decision point
solved, and it is why the stopping rule has to be NashConv and not patience.

Two ways to make that tractable:

- **Start with `--peek-memory 0`.** That drops peek memory and shrinks the tree
  to roughly the old solve's size, so it converges far sooner. Get that one
  genuinely flat first; it is already a better answer than the old file. Then
  spend the long run on the full-width version.
- **Run it under PyPy.** Same code, large speedup, no changes needed.

Checkpoints are written as it goes and `--resume` picks up exactly where it
stopped, so a long run can be done in stages.

## Files

| file | |
|---|---|
| `coup/game.py` | rules, state, all six decision phases |
| `coup/solver.py` | outcome-sampling MCCFR, regret matching+, linear averaging |
| `coup/exploit.py` | NashConv, Monte-Carlo best response, deviation check |
| `coup/export.py` | lookup and CSV export |
| `coup/kuhn.py` | the correctness test game |
| `solve_coup.py` | CLI |
| `analyze_cfr.py` | reads the *old* pickle format; kept for reference |
