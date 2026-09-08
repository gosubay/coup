#!/usr/bin/env python3
"""
Read a CFR solve of heads-up Coup (coup_gto_strategy.pkl) and emit the
aggregate policy blob embedded in index.html's Solver / GTO Strategy tabs.

    python3 tools/analyze_cfr.py coup_gto_strategy.pkl > data.json

The pickle stores only integer tags for cards and actions, and the classes
that defined them are not in the file. Both enumerations are recovered from
the structure of the solve itself -- see recover_encoding() below.
"""
import bisect
import collections
import json
import math
import pickle
import sys

CARD = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"]

# Legal action list at an action node, filtered from a fixed order. The regret
# vector indexes into the *filtered* list, so index 3 means different things at
# different coin counts.
def legal_actions(coins):
    if coins >= 10:
        return ["Coup"]
    acts = ["Income", "ForeignAid", "Tax", "Steal", "Exchange"]
    if coins >= 3:
        acts.append("Assassinate")
    if coins >= 7:
        acts.append("Coup")
    return acts

# Response options, keyed by the pending action tag.
RESPONSES = {
    1: ["Pass", "BlockFA(Duke)"],                                    # Foreign Aid
    3: ["Pass", "Challenge"],                                        # Tax
    4: ["Pass", "Challenge", "Block(Contessa)"],                     # Assassinate
    5: ["Pass", "Challenge", "Block(Captain)", "Block(Ambassador)"], # Steal
    6: ["Pass", "Challenge"],                                        # Exchange
}
BLOCK_CARD = {9: 4, 10: 2, 11: 0, 12: 3}   # block tag -> card that backs it
CLAIM_CARD = {3: 0, 4: 1, 5: 2, 6: 3}      # action tag -> card it claims
ACTION_NAME = {1: "ForeignAid", 3: "Tax", 4: "Assassinate", 5: "Steal", 6: "Exchange"}
BLOCK_NAME = {
    (1, 11): "Duke blocks Foreign Aid",
    (4, 9): "Contessa blocks Assassination",
    (5, 10): "Captain blocks Steal",
    (5, 12): "Ambassador blocks Steal",
}
BLOCK_OPTION = {9: "Block(Contessa)", 10: "Block(Captain)",
                11: "BlockFA(Duke)", 12: "Block(Ambassador)"}


class _Stub:
    """Stands in for the solver's Card / Phase / Action / Node classes."""
    def __init__(self, *args, **kw):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh):
        super().__init__(fh)
        self._made = {}

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            pass
        key = (module, name)
        if key not in self._made:
            self._made[key] = type(name, (_Stub,), {"__module__": module})
        return self._made[key]


def tag(x):
    """Enum stub -> its integer value; everything else unchanged."""
    return x._args[0] if isinstance(x, _Stub) and x._args else x


def load(path):
    """Flatten the pickle into records with named fields and named actions.

    Key layout, always from the perspective of whoever is to move:
        (me_influence, opp_influence, me_coins, opp_coins,
         my_hand, revealed, phase, pending_action, pending_block)
    """
    with open(path, "rb") as fh:
        raw = _Unpickler(fh).load()

    out = []
    for key, node in raw.items():
        phase = tag(key[6])
        pend = tag(key[7])
        strategy_sum = node.__dict__["strategy_sum"]
        total = sum(max(v, 0.0) for v in strategy_sum)
        if total > 0:
            avg = [max(v, 0.0) / total for v in strategy_sum]
        else:
            avg = [1.0 / len(strategy_sum)] * len(strategy_sum)

        if phase == 0:
            acts = legal_actions(key[2])
        elif phase == 1:
            acts = RESPONSES[pend]
        else:
            acts = ["Accept", "ChallengeBlock"]
        assert len(acts) == len(avg), (key, acts, avg)

        out.append({
            "me_lives": key[0], "opp_lives": key[1],
            "me_coins": key[2], "opp_coins": key[3],
            "hand": [tag(c) for c in key[4]],
            "rev": [tag(c) for c in key[5]],
            "phase": phase, "pend": pend, "blk": tag(key[8]),
            "acts": acts, "p": avg,
            "w": total,                       # accumulated reach mass
            "r": node.__dict__["regret_sum"],
        })
    return out


def recover_encoding(recs):
    """Re-derive the card and action tags from the solve, as a self-check.

    Cards: at a response node, the option whose frequency jumps when a given
    card is in hand is the block backed by that card -- which names both.
    Actions: the pending action with four responses can only be Steal, the one
    with three is Assassinate, and Foreign Aid is the only one whose sole
    response is a block.
    """
    problems = []
    for pend, blocks in ((1, [11]), (4, [9]), (5, [10, 12])):
        for blk in blocks:
            card = BLOCK_CARD[blk]
            opt = BLOCK_OPTION[blk]
            held, free = [0.0, 0.0], [0.0, 0.0]
            for x in recs:
                if x["phase"] != 1 or x["pend"] != pend:
                    continue
                bucket = held if card in x["hand"] else free
                bucket[0] += x["p"][x["acts"].index(opt)] * x["w"]
                bucket[1] += x["w"]
            lift = (held[0] / held[1]) / (free[0] / free[1])
            if lift < 1.5:
                problems.append(f"{opt}: holding {CARD[card]} lifts it only x{lift:.2f}")
    arity = collections.Counter(
        (x["pend"], len(x["acts"])) for x in recs if x["phase"] == 1)
    for pend, expected in ((5, 4), (4, 3), (1, 2), (3, 2), (6, 2)):
        if not any(k == (pend, expected) for k in arity):
            problems.append(f"action {pend} does not have {expected} responses")
    return problems


def mix(recs, predicate):
    """Reach-weighted action mix over the information sets matching predicate."""
    acc = collections.defaultdict(float)
    weight = 0.0
    for x in recs:
        if not predicate(x):
            continue
        weight += x["w"]
        for act, p in zip(x["acts"], x["p"]):
            acc[act] += p * x["w"]
    if not weight:
        return {}, 0.0
    return {k: round(v / weight, 4) for k, v in acc.items()}, weight


def regret_matched(regret):
    pos = [max(v, 0.0) for v in regret]
    total = sum(pos)
    return [v / total for v in pos] if total else [1.0 / len(regret)] * len(regret)


def build(recs):
    total_mass = sum(x["w"] for x in recs)
    D = {}

    D["meta"] = {
        "infosets": len(recs),
        "byPhase": {str(p): sum(1 for x in recs if x["phase"] == p) for p in (0, 1, 2)},
        "massByPhase": {str(p): round(sum(x["w"] for x in recs if x["phase"] == p))
                        for p in (0, 1, 2)},
        "totalMass": round(total_mass),
    }

    D["turnByCoins"] = {
        str(c): mix(recs, lambda x, c=c: x["phase"] == 0 and x["me_coins"] == c)[0]
        for c in range(13)
    }

    # Claim frequency with and without the card behind it, plus the share of
    # claims that turn out to be honest and the rate they get challenged.
    D["claim"] = {}
    for pend, card in CLAIM_CARD.items():
        name = ACTION_NAME[pend]
        held = [0.0, 0.0]
        free = [0.0, 0.0]
        for x in recs:
            if x["phase"] != 0 or name not in x["acts"]:
                continue
            claimed = x["p"][x["acts"].index(name)] * x["w"]
            bucket = held if card in x["hand"] else free
            bucket[0] += claimed
            bucket[1] += x["w"]
        challenged = mix(recs, lambda x, p=pend: x["phase"] == 1 and x["pend"] == p)[0]
        D["claim"][name] = {
            "card": CARD[card],
            "with_card": round(held[0] / held[1], 4),
            "without": round(free[0] / free[1], 4),
            "honest": round(held[0] / (held[0] + free[0]), 4),
            "challenged": challenged.get("Challenge", challenged.get("BlockFA(Duke)", 0)),
        }

    root = [x for x in recs if x["phase"] == 0 and x["me_lives"] == 2
            and x["opp_lives"] == 2 and x["me_coins"] == 2 and x["opp_coins"] == 2
            and not x["rev"]]
    D["root"] = {
        "acts": root[0]["acts"],
        "rows": [{"hand": [CARD[c] for c in x["hand"]],
                  "p": [round(v, 4) for v in x["p"]],
                  "w": round(x["w"])}
                 for x in sorted(root, key=lambda y: y["hand"])],
    }

    D["resp"] = {ACTION_NAME[p]: mix(recs, lambda x, p=p: x["phase"] == 1 and x["pend"] == p)[0]
                 for p in RESPONSES}

    # Card removal: the more copies of the claimed card the responder can see,
    # the less room the claim has to be honest.
    D["blockerChal"] = {}
    for pend, card in CLAIM_CARD.items():
        row = {}
        for n in range(4):
            m, w = mix(recs, lambda x, p=pend, c=card, n=n:
                       x["phase"] == 1 and x["pend"] == p
                       and x["hand"].count(c) + x["rev"].count(c) == n)
            if m:
                row[str(n)] = {"chal": m.get("Challenge", 0), "w": round(w)}
        D["blockerChal"][ACTION_NAME[pend]] = row

    D["blockRates"] = {}
    D["p2blocker"] = {}
    for (pend, blk), name in BLOCK_NAME.items():
        card = BLOCK_CARD[blk]
        opt = BLOCK_OPTION[blk]
        held = [0.0, 0.0]
        free = [0.0, 0.0]
        for x in recs:
            if x["phase"] != 1 or x["pend"] != pend:
                continue
            blocked = x["p"][x["acts"].index(opt)] * x["w"]
            bucket = held if card in x["hand"] else free
            bucket[0] += blocked
            bucket[1] += x["w"]
        challenged, _ = mix(recs, lambda x, p=pend, b=blk:
                            x["phase"] == 2 and x["pend"] == p and x["blk"] == b)
        D["blockRates"][name] = {
            "with_card": round(held[0] / held[1], 4),
            "without": round(free[0] / free[1], 4),
            "honest": round(held[0] / (held[0] + free[0]), 4),
            "challenged": challenged.get("ChallengeBlock", 0),
        }
        row = {}
        for n in range(4):
            m, w = mix(recs, lambda x, p=pend, b=blk, c=card, n=n:
                       x["phase"] == 2 and x["pend"] == p and x["blk"] == b
                       and x["hand"].count(c) + x["rev"].count(c) == n)
            if m:
                row[str(n)] = {"chal": m.get("ChallengeBlock", 0), "w": round(w)}
        D["p2blocker"][name] = row

    D["assassinDef"] = {}
    for lives in (1, 2):
        for has in (True, False):
            m, w = mix(recs, lambda x, l=lives, h=has:
                       x["phase"] == 1 and x["pend"] == 4 and x["me_lives"] == l
                       and (4 in x["hand"]) == h)
            D["assassinDef"][f"{lives}|{int(has)}"] = {"p": m, "w": round(w)}

    D["chalMatrix"] = {}
    for mine in (2, 1):
        for theirs in (2, 1):
            m, w = mix(recs, lambda x, a=mine, b=theirs:
                       x["phase"] == 1 and x["pend"] in (3, 5, 6)
                       and x["me_lives"] == a and x["opp_lives"] == b)
            D["chalMatrix"][f"{mine}|{theirs}"] = {"chal": m.get("Challenge", 0),
                                                   "w": round(w)}

    D["faByCoins"] = {
        str(c): mix(recs, lambda x, c=c: x["phase"] == 1 and x["pend"] == 1
                    and x["opp_coins"] == c)[0].get("BlockFA(Duke)", 0)
        for c in range(10)
    }
    D["stealByCoins"] = {
        str(c): mix(recs, lambda x, c=c: x["phase"] == 0 and x["opp_coins"] == c
                    and 2 in x["hand"] and "Steal" in x["acts"])[0].get("Steal", 0)
        for c in range(10)
    }
    D["coupVsAss"] = {}
    for coins in (7, 8, 9):
        for has in (True, False):
            m, _ = mix(recs, lambda x, c=coins, h=has:
                       x["phase"] == 0 and x["me_coins"] == c
                       and (1 in x["hand"]) == h)
            D["coupVsAss"][f"{coins}|{int(has)}"] = {
                k: m.get(k, 0) for k in ("Coup", "Assassinate", "Income")}

    # Convergence proxies. There is no exploitability trace in the file, so:
    # how far the current regret-matching policy has drifted from the running
    # average, how sharp the average is, and how concentrated the traffic is.
    purity = collections.Counter()
    drift = 0.0
    entropy = collections.defaultdict(lambda: [0.0, 0.0])
    for x in recs:
        top = max(x["p"])
        purity["pure" if top > .95 else "lean" if top > .7
                else "mixed" if top > .4 else "wide"] += 1
        drift += sum(abs(a - b) for a, b in zip(regret_matched(x["r"]), x["p"])) / 2 * x["w"]
        e = -sum(p * math.log(p, 2) for p in x["p"] if p > 0)
        entropy[x["phase"]][0] += e * x["w"]
        entropy[x["phase"]][1] += x["w"]

    masses = sorted(x["w"] for x in recs)
    descending = masses[::-1]
    concentration = {}
    for frac in (0.5, 0.8, 0.9, 0.99):
        acc = 0.0
        i = 0
        while acc < frac * total_mass:
            acc += descending[i]
            i += 1
        concentration[str(frac)] = i

    n = len(masses)
    D["diag"] = {
        "purity": dict(purity),
        "drift": round(drift / total_mass, 4),
        "entropy": {str(p): round(entropy[p][0] / entropy[p][1], 3) for p in (0, 1, 2)},
        "massQuantiles": {"p10": masses[int(.1 * n)], "median": masses[n // 2],
                          "p90": masses[int(.9 * n)]},
        "concentration": concentration,
        "lowMass": {str(t): bisect.bisect_left(masses, t) for t in (10, 100, 1000, 10000)},
        "rootVisits": max(x["w"] for x in root),
    }
    return D


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    recs = load(sys.argv[1])
    problems = recover_encoding(recs)
    if problems:
        print("encoding self-check failed:", *problems, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print(f"{len(recs)} information sets", file=sys.stderr)
    json.dump(build(recs), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
