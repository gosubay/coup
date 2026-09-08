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


class Solver:
    def __init__(self, game_factory, seed=0, epsilon=0.6, tag=None,
                 frozen=None, frozen_player=None):
        self.game_factory = game_factory
        # When `frozen` is set, the named player plays that fixed policy and
        # never learns. Training the other player then converges to a best
        # response -- which is how exploitability gets measured.
        self.frozen = frozen
        self.frozen_player = frozen_player
        self.nodes: dict = {}          # key -> [regrets, strategy_sum]
        self.actions: dict = {}        # key -> legal action tuple
        self.rng = random.Random(seed)
        self.epsilon = epsilon         # exploration in the sampling policy
        self.t = 0
        self.trace = []
        self.tag = tag or {}

    # -------------------------------------------------------------- nodes --
    def _node(self, key, legal):
        n = self.nodes.get(key)
        if n is None:
            k = len(legal)
            n = [[0.0] * k, [0.0] * k]
            self.nodes[key] = n
            self.actions[key] = legal
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
            u, tail = self._os(st.apply_chance(card), i, pi_i, pi_o, samp * p, weight)
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
            regrets, strat = self._node(key, legal)
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

    # -------------------------------------------------------------- drive --
    def run(self, iterations, report_every=0, checkpoint=None, checkpoint_every=0,
            exploit_every=0, exploit_fn=None):
        start = time.time()
        for _ in range(iterations):
            self.t += 1
            w = float(self.t)
            for i in (0, 1):
                self._os(self.game_factory(), i, 1.0, 1.0, 1.0, w)

            if report_every and self.t % report_every == 0:
                el = time.time() - start
                print(f"  t={self.t:>10,}  infosets={len(self.nodes):>9,}  "
                      f"{self.t/el:9.0f} it/s", flush=True)

            if exploit_every and exploit_fn and self.t % exploit_every == 0:
                nc = exploit_fn(self)
                self.trace.append((self.t, nc, time.time() - start))
                print(f"  t={self.t:>10,}  NashConv={nc:+.5f}  "
                      f"<- stop when this flattens", flush=True)

            if checkpoint and checkpoint_every and self.t % checkpoint_every == 0:
                self.save(checkpoint)
        if checkpoint:
            self.save(checkpoint)

    # ----------------------------------------------------------- policies --
    def average_strategy(self):
        out = {}
        for key, (_, strat) in self.nodes.items():
            tot = sum(strat)
            k = len(strat)
            out[key] = (self.actions[key],
                        [s / tot for s in strat] if tot > 0 else [1.0 / k] * k)
        return out

    def current_strategy(self):
        """Diagnostic only. This one never settles -- that is normal."""
        return {key: (self.actions[key], self._strategy(r))
                for key, (r, _) in self.nodes.items()}

    # --------------------------------------------------------------- i/o --
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"version": 3, "iterations": self.t, "trace": self.trace,
                         "actions": self.actions, "nodes": self.nodes,
                         "tag": self.tag, "epsilon": self.epsilon},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path, game_factory=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        s = cls(game_factory, epsilon=d.get("epsilon", 0.6), tag=d.get("tag", {}))
        s.t = d["iterations"]; s.trace = d["trace"]
        s.actions = d["actions"]; s.nodes = d["nodes"]
        return s


def policy_stats(policy):
    n = len(policy)
    pure = sum(1 for _, p in policy.values() if max(p) > 0.99)
    near = sum(1 for _, p in policy.values() if 0.90 < max(p) <= 0.99)
    return {"infosets": n, "pure": pure, "near_pure": near, "mixed": n - pure - near}
