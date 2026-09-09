# Coup solver

A heads-up Coup solver that reports whether its own answer is finished.

The previous solve did not, and that turned out to be the whole problem: a
five-line greedy heuristic beat it by 17 percentage points of win rate, while
its own stored regrets already said "Coup more" and the average strategy had
not caught up.

## Running it on your own machine

No dependencies -- it is standard library only, so any Python 3.8+ will do.

The solver lives on the `claude/cfr-sims-analysis-strategy-mebgh9` branch, not
on `main`, so clone that branch or you will get a repository with no solver in
it. Do it somewhere you can write -- not `C:\Windows\System32`.

**Windows (PowerShell):**

```powershell
cd $HOME
git clone -b claude/cfr-sims-analysis-strategy-mebgh9 https://github.com/gosubay/Coup.git
cd Coup
python tools\solve_coup.py bench      # measure YOUR machine, ~1 minute
python tools\run.py                   # run the plan
```

**macOS / Linux:**

```bash
cd ~
git clone -b claude/cfr-sims-analysis-strategy-mebgh9 https://github.com/gosubay/Coup.git
cd Coup
python3 tools/solve_coup.py bench
python3 tools/run.py                  # or tools/run.sh start to detach
```

`run.py` is the cross-platform driver and runs in the foreground; Ctrl+C is
safe, because every stage checkpoints and re-running resumes. `run.sh` does the
same thing detached, but it is a bash script and will not run under PowerShell.

Useful flags:

```
python tools/run.py --iters 200000    # a few-minute trial before committing hours
python tools/run.py --only base       # just the one configuration that converges
python tools/run.py --status          # how far along, and is it flat yet
```

`bench` times all three configurations on your hardware and projects the wall
time and peak RAM for the whole plan, so you get real numbers instead of the
ones measured here.

### PyPy

The solver is one tight pure-Python loop, which is the case PyPy is built for.
It runs the same code with no changes and no separate build:

```bash
# macOS: brew install pypy3   |   Debian/Ubuntu: apt install pypy3
# or download from https://pypy.org/download.html
pypy3 tools/solve_coup.py bench       # compare this against the python3 number
pypy3 tools/solve_coup.py solve --peek-memory 0 --claim-memory 0 --iters 20000000
```

Run `bench` under both and use whichever is faster -- the checkpoint format is
identical, so you can even solve under PyPy and query under CPython.

Two things PyPy does not change: the tree sizes and the bytes per infoset. If
`full` does not fit in your RAM under CPython, it will not fit under PyPy
either. Memory is the binding constraint on the big configurations; speed is
the binding constraint on the small ones.

## Quick start

One command runs the whole plan, smallest tree first, and survives logout:

```bash
tools/run.sh start      # launch everything detached
tools/run.sh status     # where each stage is, and whether it has flattened
tools/run.sh stop       # halt; checkpoints are kept, `start` resumes
```

It runs three stages into `solves/`, each one a superset of the last. All
figures below are measured on one core of CPython 3.11, not guessed:

| stage | peek | claim | tree | ceiling | wall time | RAM | avg visits/infoset |
|---|---|---|---|---|---|---|---|
| `base` | 0 | 0 | ~380k, saturated | 20M iters | ~1.3 h | 0.2 GB | ~1,400 |
| `claims` | 0 | 1 | ~6M | 60M iters | ~5.5 h | 2.6 GB | ~270 |
| `full` | 1 | 1 | 15-30M, still growing | 150M iters | ~13 h | 7-13 GB | ~200 |

About 20 hours end to end. Each traversal touches 27 infosets, so iterations
buy visits at 27:1 -- but visits are reach-weighted, so a rare infoset gets far
fewer than the average and a common one far more.

**Only `base` will finish properly.** Its tree stops growing before 2M
iterations, so 20M iterations genuinely converge it. `claims` and `full` are
still discovering new infosets when the budget runs out; their NashConv will
still be falling. That is a real answer -- just an unfinished one, and `status`
will say so rather than pretending otherwise.

The iteration counts are a ceiling, not a target. **Stop a stage when its
NashConv stops falling**, not when it hits the number -- that is what `status`
reports. Every stage checkpoints and resumes, so interrupting costs nothing.

For a quick trial before committing hours:

```bash
COUP_STAGES="trial 0 0 200000" tools/run.sh start
```

Or drive the pieces yourself:

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
| **roles each side has claimed** | **new** -- this is the bluff-consistency information |
| what you are responding to | pending action, pending block |

Claims are public, so both masks appear in both players' keys. A claim expires
when the hand behind it can have changed: an Exchange clears that player's whole
mask, and winning a challenge clears only the bit for the card that was shuffled
back. Under random play 21% of live states already have a player claiming more
distinct roles than they hold influences -- a proven lie the old solve could not
see.

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
4. **Claim memory.** Which roles each side has claimed since their hand last
   changed. Without it the solver re-meets an opponent who has claimed Duke,
   Captain and Assassin on two influences as if they had claimed nothing, so it
   cannot punish a bluffing line and cannot price the cost of its own. Run with
   `--claim-memory 0` to turn it off.

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

- **Turn the memories on one at a time.** Each one multiplies the tree. Measured
  at a fixed 60,000 iterations:

  | peek | claim | infosets found | it/s |
  |---|---|---|---|
  | 0 | 0 | 140,130 | 3,916 |
  | 1 | 0 | 309,678 | 3,802 |
  | 0 | 1 | 491,281 | 3,010 |
  | 1 | 1 | 610,722 | 3,233 |

  Claim memory is the more expensive of the two and also the more valuable.

- **Memory is the binding constraint, not time.** An infoset costs 443 bytes:
  the key is packed into a single int, regrets and strategy sums are `array('d')`
  rather than lists of boxed floats, and identical legal-action tuples are
  shared. That is down from 780 bytes and about 10% faster, because the two
  parallel dicts became one. At `full` scale it is still the difference between
  9 GB and 16 GB. Pass `--max-gb` (`run.sh` sets it to two thirds of RAM) and a
  solve that reaches the limit checkpoints and stops cleanly instead of being
  OOM-killed hours in.

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
