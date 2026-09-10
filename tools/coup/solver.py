"""Outcome-sampling MCCFR with regret matching+ and linear averaging.

Two deliberate choices, both fixes for what went wrong last time:

  regret matching+   cumulative regrets are clamped at zero, so a bad early
                     stretch stops dragging the answer around forever.

  linear averaging   iteration t contributes with weight t, so recent and
                     better iterations outvote ancient ones instead of the
                     other way round. This is the specific defect that left
                     the previous solve coupling far too rarely.

Outcome sampling (one trajectory per traversal) rather than external sampling,
because external sampling enumerates every action of the player being updated
and Coup games are long enough that this explodes combinatorially.

The algorithm is game-agnostic: any state object providing is_terminal,
utility, is_chance, chance_outcomes, current_player, legal, infoset_key, apply
and apply_chance will do. See kuhn.py for the correctness test.
"""

from __future__ import annotations

import pickle
import random
import time
from array import array


class Solver:
    def __init__(self, game_factory, seed=0, epsilon=0.6, tag=None,
                 frozen=None, frozen_player=None, sampling="robust", rs_k=2):
        self.game_factory = game_factory
        # When `frozen` is set, the named player plays that fixed policy and
        # never learns. Training the other player then converges to a best
        # response -- which is how exploitability gets measured.
        self.frozen = frozen
        self.frozen_player = frozen_player
        # one dict, not two: a second dict keyed on the same infosets costs
        # another ~100 bytes each, and there are tens of millions of them
        self.nodes: dict = {}          # key -> [regrets, strategy_sum, legal]
        # every infoset with the same legal actions shares one tuple
        self._legal: dict = {}
        self.rng = random.Random(seed)
        self.epsilon = epsilon         # exploration in the sampling policy
        self.t = 0
        self.trace = []
        self.tag = tag or {}
        # "external": enumerate the traverser's actions, sample chance and the
        # opponent. No importance weights anywhere, so nothing can explode.
        # "outcome": one trajectory, importance-weighted -- cheap per iteration
        # but the weights reach 1e17 at Coup's depth and freeze the policy on
        # early noise.
        self.sampling = sampling
        # robust sampling walks min(rs_k, |A|) of the traverser's actions. k>=|A|
        # is exactly external sampling; k=1 collapses towards outcome sampling.
        # The importance weight is bounded by (|A|/k) per traverser node instead
        # of (|A|/epsilon), which is the whole point.
        self.rs_k = rs_k

    # -------------------------------------------------------------- nodes --
    def _node(self, key, legal):
        n = self.nodes.get(key)
        if n is None:
            k = len(legal)
            # array('d') rather than a list: a list of k floats is k boxed float
            # objects, which at this scale is most of the memory in the process
            n = [array("d", bytes(8 * k)), array("d", bytes(8 * k)),
                 self._legal.setdefault(legal, legal)]
            self.nodes[key] = n
        return n

    @staticmethod
    def _strategy(regrets):
        tot = 0.0
        for r in regrets:
            tot += r
        if tot <= 0.0:
            u = 1.0 / len(regrets)
            return [u] * len(regrets)
        return [r / tot for r in regrets]

    def _pick(self, probs):
        x = self.rng.random(); acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if x < acc:
                return i
        return len(probs) - 1

    # ---------------------------------------------------------- traversal --
    def _os(self, st, i, pi_i, pi_o, samp, weight):
        """Outcome sampling.

        Returns (u_i(z)/q(z), pi^sigma(this node -> z)).
        pi_i / pi_o are player i's / everyone else's reach under sigma;
        `samp` is the probability of having sampled the path to here.
        """
        if st.is_terminal():
            return st.utility(i) / samp, 1.0

        if st.is_chance():
            out = st.chance_outcomes()
            j = self._pick([p for _, p in out])
            card, p = out[j]
            # chance belongs in the counterfactual reach: pi_o is everything that
            # is not the traverser. Leaving it out while 1/q still divides by it
            # inflates every line reached through an improbable deal by the
            # reciprocal of that deal's probability. Invisible on a game whose
            # chance is all at the root, fatal on one that interleaves it.
            u, tail = self._os(st.apply_chance(card), i, pi_i, pi_o * p,
                               samp * p, weight)
            return u, tail * p

        p = st.current_player()
        legal = st.legal()
        key = st.infoset_key(p)
        k = len(legal)

        if self.frozen is not None and p == self.frozen_player:
            e = self.frozen.get(key)
            sigma = e[1] if e is not None else [1.0 / k] * k
            regrets = strat = None
        else:
            regrets, strat, _ = self._node(key, legal)
            sigma = self._strategy(regrets)

        if p == i:
            e = self.epsilon
            rho = [e / k + (1.0 - e) * sigma[j] for j in range(k)]
        else:
            rho = sigma

        a = self._pick(rho)
        child = st.apply(legal[a])

        if p == i:
            u, tail = self._os(child, i, pi_i * sigma[a], pi_o, samp * rho[a], weight)
            # counterfactual value of the one action we actually sampled
            v = pi_o * tail * u
            sa = sigma[a]
            for j in range(k):
                d = v * ((1.0 if j == a else 0.0) - sa)
                r = regrets[j] + d
                regrets[j] = r if r > 0.0 else 0.0
            w = weight * pi_i / samp          # linear averaging, importance corrected
            for j in range(k):
                strat[j] += w * sigma[j]
        else:
            u, tail = self._os(child, i, pi_i, pi_o * sigma[a], samp * rho[a], weight)

        return u, tail * sigma[a]

    # ---------------------------------------------------- external sampling --
    def _es(self, st, i, weight):
        """External-sampling MCCFR. Returns the sampled counterfactual value.

        The traverser's every action is walked, so its regrets carry no sampling
        weight at all. Chance and the opponent are sampled from their own
        distributions, which means the frequency with which an opponent infoset
        is visited is already proportional to its reach -- so the average
        strategy accumulates on-policy, with no 1/q correction to blow up.
        """
        if st.is_terminal():
            return st.utility(i)

        if st.is_chance():
            out = st.chance_outcomes()
            card, _p = out[self._pick([p for _, p in out])]
            return self._es(st.apply_chance(card), i, weight)

        p = st.current_player()
        legal = st.legal()
        k = len(legal)

        if self.frozen is not None and p == self.frozen_player:
            e = self.frozen.get(st.infoset_key(p))
            sigma = e[1] if e is not None else [1.0 / k] * k
            return self._es(st.apply(legal[self._pick(sigma)]), i, weight)

        regrets, strat, _ = self._node(st.infoset_key(p), legal)
        sigma = self._strategy(regrets)

        if p != i:
            # linear averaging: iteration t counts for t
            for j in range(k):
                strat[j] += weight * sigma[j]
            return self._es(st.apply(legal[self._pick(sigma)]), i, weight)

        kk = self.rs_k if self.sampling == "robust" else k
        if kk >= k:
            node_util = 0.0
            util = [0.0] * k
            for j in range(k):
                util[j] = self._es(st.apply(legal[j]), i, weight)
                node_util += sigma[j] * util[j]
            for j in range(k):
                r = regrets[j] + util[j] - node_util
                regrets[j] = r if r > 0.0 else 0.0    # regret matching+
            return node_util

        # walk kk of the k actions, uniformly and without replacement. Each is
        # sampled with probability kk/k, so scaling its value by k/kk keeps the
        # estimate unbiased; an unsampled action still gets its -v(I) term.
        picks = self.rng.sample(range(k), kk)
        scale = k / kk
        util = {}
        node_util = 0.0
        for j in picks:
            u = self._es(st.apply(legal[j]), i, weight)
            util[j] = u
            node_util += sigma[j] * scale * u
        for j in range(k):
            adv = (scale * util[j] - node_util) if j in util else -node_util
            r = regrets[j] + adv
            regrets[j] = r if r > 0.0 else 0.0
        return node_util

    # -------------------------------------------------------------- drive --
    @staticmethod
    def _rss_gb():
        """Resident memory, or 0.0 where /proc is not available."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1048576.0
        except OSError:
            pass
        return 0.0

    def run(self, iterations, report_every=0, checkpoint=None, checkpoint_every=0,
            exploit_every=0, exploit_fn=None, max_gb=0.0):
        start = time.time()
        for _ in range(iterations):
            self.t += 1
            w = float(self.t)
            if self.sampling in ("external", "robust"):
                for i in (0, 1):
                    self._es(self.game_factory(), i, w)
            else:
                for i in (0, 1):
                    self._os(self.game_factory(), i, 1.0, 1.0, 1.0, w)

            if report_every and self.t % report_every == 0:
                el = time.time() - start
                print(f"  t={self.t:>10,}  infosets={len(self.nodes):>9,}  "
                      f"{self.t/el:9.0f} it/s", flush=True)

            if exploit_every and exploit_fn and self.t % exploit_every == 0:
                nc, budget = exploit_fn(self)
                self.trace.append((self.t, nc, time.time() - start))
                # The budget is part of the reading: two NashConv numbers are
                # only comparable if the responder behind them was trained
                # comparably. Without it the trace is not a convergence signal.
                print(f"  t={self.t:>10,}  NashConv>={nc:.5f}  "
                      f"(responder budget {budget:,})", flush=True)

            if checkpoint and checkpoint_every and self.t % checkpoint_every == 0:
                self.save(checkpoint)
                # stop on our own terms rather than being OOM-killed, which
                # would lose everything since the last checkpoint
                if max_gb and self._rss_gb() > max_gb:
                    print(f"  stopping at t={self.t:,}: {self._rss_gb():.1f} GB "
                          f"resident, over the {max_gb:.1f} GB budget. "
                          f"Checkpoint saved. Solve a smaller configuration, or "
                          f"raise --max-gb if the machine has the memory.",
                          flush=True)
                    break
        if checkpoint:
            self.save(checkpoint)

    # ----------------------------------------------------------- policies --
    def average_strategy(self):
        out = {}
        for key, (_, strat, legal) in self.nodes.items():
            tot = sum(strat)
            k = len(strat)
            out[key] = (legal,
                        [s / tot for s in strat] if tot > 0 else [1.0 / k] * k)
        return out

    def current_strategy(self):
        """Diagnostic only. This one never settles -- that is normal."""
        return {key: (legal, self._strategy(r))
                for key, (r, _, legal) in self.nodes.items()}

    # --------------------------------------------------------------- i/o --
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"version": 4, "iterations": self.t, "trace": self.trace,
                         "nodes": self.nodes, "sampling": self.sampling,
                         "rs_k": self.rs_k,
                         "tag": self.tag, "epsilon": self.epsilon},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path, game_factory=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        s = cls(game_factory, epsilon=d.get("epsilon", 0.6), tag=d.get("tag", {}),
                sampling=d.get("sampling", "outcome"), rs_k=d.get("rs_k", 3))
        if d.get("version", 0) < 4:
            raise SystemExit(f"{path} was written by an older, incompatible "
                             "format. Delete it and solve again.")
        s.t = d["iterations"]; s.trace = d["trace"]
        s.nodes = d["nodes"]
        return s


def policy_stats(policy):
    n = len(policy)
    pure = sum(1 for _, p in policy.values() if max(p) > 0.99)
    near = sum(1 for _, p in policy.values() if 0.90 < max(p) <= 0.99)
    return {"infosets": n, "pure": pure, "near_pure": near, "mixed": n - pure - near}
