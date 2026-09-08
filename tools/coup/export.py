"""Turn a solved policy into something you can actually look things up in."""

from __future__ import annotations

import csv

from .game import (CARD_NAMES, ACTION_NAMES, RESPONSE_NAMES, BLOCK_RESPONSE_NAMES,
                   PHASE_ACTION, PHASE_RESPOND, PHASE_BLOCK_RESP,
                   PHASE_DISCARD, PHASE_EXCHANGE)

CARD_ID = {n.lower(): i for i, n in enumerate(CARD_NAMES)}
ACTION_ID = {n.lower(): i for i, n in enumerate(ACTION_NAMES)}
PHASE_NAMES = {PHASE_ACTION: "action", PHASE_RESPOND: "respond",
               PHASE_BLOCK_RESP: "answer-block", PHASE_DISCARD: "discard",
               PHASE_EXCHANGE: "exchange-keep"}


def parse_cards(text):
    """'duke,contessa' or 'Duke+Contessa' -> (0, 4) sorted."""
    if not text:
        return ()
    parts = [p.strip().lower() for p in text.replace("+", ",").split(",") if p.strip()]
    return tuple(sorted(CARD_ID[p] for p in parts))


def action_label(phase, action):
    if phase == PHASE_ACTION:
        return ACTION_NAMES[action]
    if phase == PHASE_RESPOND:
        return RESPONSE_NAMES[action]
    if phase == PHASE_BLOCK_RESP:
        return BLOCK_RESPONSE_NAMES[action]
    if phase == PHASE_DISCARD:
        return f"reveal {CARD_NAMES[action]}"
    if phase == PHASE_EXCHANGE:
        return "keep " + "+".join(CARD_NAMES[c] for c in action)
    return str(action)


def make_key(my_lives, opp_lives, my_coins, opp_coins, my_hand,
             revealed=(), peek=(), phase=PHASE_ACTION, pend=None, blk=None,
             pool=None):
    """Build the infoset key exactly as the engine does."""
    tail = tuple(sorted(pool)) if phase == PHASE_EXCHANGE else tuple(sorted(my_hand))
    return (my_lives, opp_lives, my_coins, opp_coins,
            tuple(sorted(revealed)), tuple(sorted(peek)),
            phase, pend, blk, tail)


def lookup(policy, **kw):
    """-> [(label, probability)] sorted by probability, or None if unreached."""
    key = make_key(**kw)
    e = policy.get(key)
    if e is None:
        return None
    actions, probs = e
    phase = key[6]
    rows = [(action_label(phase, a), p) for a, p in zip(actions, probs)]
    rows.sort(key=lambda r: -r[1])
    return rows


def export_csv(policy, path, phase=None, min_prob=0.0):
    """Flat table. One row per infoset per action."""
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["my_lives", "opp_lives", "my_coins", "opp_coins", "my_hand",
                    "face_up", "peek", "phase", "vs_action", "vs_block",
                    "choice", "frequency"])
        for key, (actions, probs) in policy.items():
            (ml, ol, mc, oc, rev, peek, ph, pend, blk, tail) = key
            if phase is not None and ph != phase:
                continue
            for a, p in zip(actions, probs):
                if p < min_prob:
                    continue
                w.writerow([
                    ml, ol, mc, oc,
                    "+".join(CARD_NAMES[c] for c in tail),
                    "+".join(CARD_NAMES[c] for c in rev),
                    "+".join(CARD_NAMES[c] for c in peek),
                    PHASE_NAMES.get(ph, ph),
                    ACTION_NAMES[pend] if pend is not None else "",
                    RESPONSE_NAMES[blk] if blk is not None else "",
                    action_label(ph, a), round(p, 6),
                ])
                n += 1
    return n
