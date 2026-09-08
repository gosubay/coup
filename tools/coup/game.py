"""Heads-up Coup: complete game engine for CFR.

Every decision a player makes is a real decision node, including the two the
previous solve automated:

  * which cards to keep after an Exchange  (PHASE_EXCHANGE)
  * which influence to reveal when you lose one  (PHASE_DISCARD)

and the information the previous solve discarded:

  * the cards you saw during an Exchange and handed back to the deck (`peek`),
    which is what makes card-removal reasoning possible.
  * which roles each player has *claimed* since their hand last changed
    (`claims`), which is what makes bluff-consistency reasoning possible: two
    influences can back at most two distinct claims, so a third one is a proven
    lie. Claims are public, so both players see both masks.

Deck is tracked as a multiset of counts, never as a shuffled list, so a draw is
an explicit chance node with exact probabilities rather than a sampled ordering.
"""

from __future__ import annotations

# ---------------------------------------------------------------- cards ----
DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA = range(5)
CARD_NAMES = ("Duke", "Assassin", "Captain", "Ambassador", "Contessa")
N_CARDS = 5
COPIES = 3
DECK_SIZE = N_CARDS * COPIES

# -------------------------------------------------------------- actions ----
INCOME, FOREIGN_AID, COUP, TAX, ASSASSINATE, STEAL, EXCHANGE = range(7)
ACTION_NAMES = ("Income", "ForeignAid", "Coup", "Tax", "Assassinate", "Steal", "Exchange")

# action -> card it claims (absent = no claim, cannot be challenged)
CLAIMS = {TAX: DUKE, ASSASSINATE: ASSASSIN, STEAL: CAPTAIN, EXCHANGE: AMBASSADOR}
# actions that can be answered at all
CHALLENGEABLE = frozenset(CLAIMS)

ASSASSIN_COST = 3
COUP_COST = 7
FORCED_COUP_AT = 10

# ------------------------------------------------------------- responses ----
PASS, CHALLENGE = 0, 1
BLOCK_DUKE, BLOCK_CONTESSA, BLOCK_CAPTAIN, BLOCK_AMBASSADOR = 2, 3, 4, 5
RESPONSE_NAMES = {
    PASS: "Pass", CHALLENGE: "Challenge",
    BLOCK_DUKE: "Block(Duke)", BLOCK_CONTESSA: "Block(Contessa)",
    BLOCK_CAPTAIN: "Block(Captain)", BLOCK_AMBASSADOR: "Block(Ambassador)",
}
BLOCK_CARD = {BLOCK_DUKE: DUKE, BLOCK_CONTESSA: CONTESSA,
              BLOCK_CAPTAIN: CAPTAIN, BLOCK_AMBASSADOR: AMBASSADOR}


def claim_cards(mask):
    """Bitmask of claimed roles -> tuple of card ids."""
    return tuple(c for c in range(N_CARDS) if mask >> c & 1)

RESPONSES = {
    FOREIGN_AID:  (PASS, BLOCK_DUKE),
    TAX:          (PASS, CHALLENGE),
    EXCHANGE:     (PASS, CHALLENGE),
    ASSASSINATE:  (PASS, CHALLENGE, BLOCK_CONTESSA),
    STEAL:        (PASS, CHALLENGE, BLOCK_CAPTAIN, BLOCK_AMBASSADOR),
}

ACCEPT, CHALLENGE_BLOCK = 0, 1
BLOCK_RESPONSE_NAMES = {ACCEPT: "Accept", CHALLENGE_BLOCK: "ChallengeBlock"}

# --------------------------------------------------------------- phases ----
PHASE_CHANCE, PHASE_ACTION, PHASE_RESPOND, PHASE_BLOCK_RESP, \
    PHASE_DISCARD, PHASE_EXCHANGE = range(6)

# chance kinds
DEAL, REDRAW, EXDRAW = 0, 1, 2


def legal_actions(coins: int):
    if coins >= FORCED_COUP_AT:
        return (COUP,)
    acts = [INCOME, FOREIGN_AID, TAX, STEAL, EXCHANGE]
    if coins >= ASSASSIN_COST:
        acts.append(ASSASSINATE)
    if coins >= COUP_COST:
        acts.append(COUP)
    return tuple(acts)


def _sub(counts, card):
    l = list(counts); l[card] -= 1; return tuple(l)


def _add(counts, card):
    l = list(counts); l[card] += 1; return tuple(l)


class State:
    """One node of the game. Mutable; always clone() before branching."""

    __slots__ = ("hands", "coins", "revealed", "deck", "peek", "to_move",
                 "phase", "pend", "blk", "pool", "stack", "turn", "max_turns",
                 "chance", "peek_memory", "claims", "claim_memory")

    def __init__(self, max_turns=60, peek_memory=1, claim_memory=1):
        self.hands = [(), ()]
        self.coins = [2, 2]
        self.revealed = ()
        self.deck = (COPIES,) * N_CARDS
        self.peek = [(), ()]          # cards this player handed back to the deck
        self.claims = [0, 0]          # roles each player has claimed, as a bitmask
        self.to_move = 0
        self.phase = PHASE_CHANCE
        self.chance = (DEAL, 0)       # (kind, arg)
        self.pend = None
        self.blk = None
        self.pool = None              # exchange pool while choosing what to keep
        self.stack = ()
        self.turn = 0
        self.max_turns = max_turns
        self.peek_memory = peek_memory
        self.claim_memory = claim_memory

    # ------------------------------------------------------------ basics --
    def clone(self):
        s = State.__new__(State)
        s.hands = list(self.hands); s.coins = list(self.coins)
        s.revealed = self.revealed; s.deck = self.deck
        s.peek = list(self.peek); s.claims = list(self.claims)
        s.to_move = self.to_move
        s.phase = self.phase; s.chance = self.chance
        s.pend = self.pend; s.blk = self.blk; s.pool = self.pool
        s.stack = self.stack; s.turn = self.turn
        s.max_turns = self.max_turns; s.peek_memory = self.peek_memory
        s.claim_memory = self.claim_memory
        return s

    def is_terminal(self):
        if self.phase == PHASE_CHANCE and self.chance is not None and self.chance[0] == DEAL:
            return False                       # still dealing; hands are legitimately empty
        return (not self.hands[0]) or (not self.hands[1]) or self.turn >= self.max_turns

    def utility(self, player):
        """+1 win, -1 loss, 0 draw."""
        a, b = len(self.hands[player]), len(self.hands[1 - player])
        if a and not b: return 1.0
        if b and not a: return -1.0
        return 0.0

    def is_chance(self):
        return self.phase == PHASE_CHANCE

    def census(self):
        """Count every copy of every card. Must always equal COPIES per card.

        During an Exchange the drawn cards live in `pool` alongside a copy of the
        owner's hand, so they are counted once here and not twice.
        """
        from collections import Counter
        n = Counter(self.hands[0]) + Counter(self.hands[1]) + Counter(self.revealed)
        if self.pool is not None:
            owner = (self.chance[1] if self.phase == PHASE_CHANCE and self.chance
                     and self.chance[0] == EXDRAW else self.to_move)
            extra = Counter(self.pool) - Counter(self.hands[owner])
            n += extra
        return tuple(n[c] + self.deck[c] for c in range(N_CARDS))

    def current_player(self):
        return self.to_move

    # ------------------------------------------------------------ chance --
    def chance_outcomes(self):
        """[(card, probability)] for the pending draw."""
        tot = sum(self.deck)
        return [(c, self.deck[c] / tot) for c in range(N_CARDS) if self.deck[c]]

    # ----------------------------------------------------------- actions --
    def legal(self):
        ph = self.phase
        if ph == PHASE_ACTION:
            return legal_actions(self.coins[self.to_move])
        if ph == PHASE_RESPOND:
            return RESPONSES[self.pend]
        if ph == PHASE_BLOCK_RESP:
            return (ACCEPT, CHALLENGE_BLOCK)
        if ph == PHASE_DISCARD:
            return tuple(sorted(set(self.hands[self.to_move])))
        if ph == PHASE_EXCHANGE:
            return self._keep_options()
        raise AssertionError("no actions at a chance node")

    def _keep_options(self):
        """Distinct multisets of size len(hand) drawn from the pool."""
        from itertools import combinations
        n = len(self.hands[self.to_move])
        return tuple(sorted(set(combinations(sorted(self.pool), n))))

    # ---------------------------------------------------------- infosets --
    def infoset_key(self, player):
        """Everything `player` legitimately knows, and nothing else."""
        opp = 1 - player
        me_h = self.hands[player]
        base = (len(me_h), len(self.hands[opp]),
                self.coins[player], self.coins[opp],
                self.revealed, self.peek[player],
                self.claims[player], self.claims[opp],
                self.phase, self.pend, self.blk)
        if self.phase == PHASE_EXCHANGE:
            return base + (tuple(sorted(self.pool)),)
        return base + (me_h,)

    # ------------------------------------------------------------- apply --
    def apply(self, action):
        s = self.clone()
        s._apply_inplace(action)
        s._advance()
        return s

    def apply_chance(self, card):
        s = self.clone()
        s._apply_chance_inplace(card)
        s._advance()
        return s

    # ------------------------------------------------------ internals ----
    def _draw(self, player, card):
        self.deck = _sub(self.deck, card)
        self.hands[player] = tuple(sorted(self.hands[player] + (card,)))

    def _return(self, player, card):
        h = list(self.hands[player]); h.remove(card)
        self.hands[player] = tuple(h)
        self.deck = _add(self.deck, card)

    def _apply_chance_inplace(self, card):
        kind, arg = self.chance
        if kind == DEAL:
            # arg counts cards dealt so far: 0,1 -> player 0 ; 2,3 -> player 1
            self._draw(0 if arg < 2 else 1, card)
            if arg < 3:
                self.chance = (DEAL, arg + 1)
                return                        # stay at chance
            self.phase = PHASE_ACTION
            self.to_move = 0
            self.chance = None
        elif kind == REDRAW:
            self._draw(arg, card)
            self.phase = None
            self.chance = None
        elif kind == EXDRAW:
            player = arg
            self.deck = _sub(self.deck, card)
            self.pool = self.pool + (card,)
            if len(self.pool) < len(self.hands[player]) + 2 and sum(self.deck) > 0:
                self.chance = (EXDRAW, player)
                return
            self.phase = PHASE_EXCHANGE
            self.to_move = player
            self.chance = None

    def _apply_inplace(self, action):
        ph = self.phase
        if ph == PHASE_ACTION:
            self._do_action(action)
        elif ph == PHASE_RESPOND:
            self._do_response(action)
        elif ph == PHASE_BLOCK_RESP:
            self._do_block_response(action)
        elif ph == PHASE_DISCARD:
            self._do_discard(action)
        elif ph == PHASE_EXCHANGE:
            self._do_exchange_keep(action)
        else:
            raise AssertionError(ph)

    # -- action -----------------------------------------------------------
    def _do_action(self, a):
        me = self.to_move; opp = 1 - me
        self.phase = None
        if a == INCOME:
            self.coins[me] += 1
            self.stack = (("end",),)
            return
        if a == COUP:
            self.coins[me] -= COUP_COST
            self.stack = (("end",), ("lose", opp))
            return
        if a == ASSASSINATE:
            self.coins[me] -= ASSASSIN_COST      # paid on declaration, never refunded
        if self.claim_memory and a in CLAIMS:   # Foreign Aid claims nothing
            self.claims[me] |= 1 << CLAIMS[a]
        self.pend = a
        self.stack = (("end",),)
        self.phase = PHASE_RESPOND
        self.to_move = opp

    # -- response to an action -------------------------------------------
    def _do_response(self, r):
        responder = self.to_move; actor = 1 - responder
        a = self.pend
        self.phase = None
        if r == PASS:
            self.stack = self.stack + (("resolve", actor, a),)
            return
        if r == CHALLENGE:
            need = CLAIMS[a]
            if need in self.hands[actor]:
                # challenge fails: challenger burns an influence, actor redraws
                self.stack = self.stack + (("resolve", actor, a),
                                           ("redraw", actor, need),
                                           ("lose", responder))
            else:
                self.stack = self.stack + (("lose", actor),)
            return
        # a block was claimed; the actor decides whether to challenge it
        if self.claim_memory:
            self.claims[responder] |= 1 << BLOCK_CARD[r]
        self.blk = r
        self.phase = PHASE_BLOCK_RESP
        self.to_move = actor

    # -- actor's answer to a block ---------------------------------------
    def _do_block_response(self, br):
        actor = self.to_move; blocker = 1 - actor
        self.phase = None
        if br == ACCEPT:
            return                                   # action is blocked, stack unwinds
        card = BLOCK_CARD[self.blk]
        if card in self.hands[blocker]:
            # block was honest: actor pays, blocker redraws, action stays blocked
            self.stack = self.stack + (("redraw", blocker, card), ("lose", actor))
        else:
            # block was a bluff: blocker pays and the action goes through anyway
            self.stack = self.stack + (("resolve", actor, self.pend), ("lose", blocker))

    # -- losing an influence ---------------------------------------------
    def _do_discard(self, card):
        p = self.to_move
        h = list(self.hands[p]); h.remove(card)
        self.hands[p] = tuple(h)
        self.revealed = tuple(sorted(self.revealed + (card,)))
        self.phase = None

    # -- keeping cards after an exchange ---------------------------------
    def _do_exchange_keep(self, keep):
        p = self.to_move
        pool = list(self.pool)
        for c in keep:
            pool.remove(c)
        returned = tuple(sorted(pool))
        for c in returned:
            self.deck = _add(self.deck, c)
        self.hands[p] = tuple(sorted(keep))
        # this is the card-removal information the old solve threw away
        if self.peek_memory:
            self.peek[p] = returned
        if self.claim_memory:
            self.claims[p] = 0        # both cards may have changed; nothing binds
        self.pool = None
        self.phase = None

    # -- run the continuation stack until someone must decide -------------
    def _advance(self):
        while self.phase is None:
            if self.is_terminal():
                return
            if not self.stack:
                self.phase = PHASE_ACTION
                return
            op = self.stack[-1]; self.stack = self.stack[:-1]
            kind = op[0]

            if kind == "end":
                self.turn += 1
                self.pend = None; self.blk = None
                if self.is_terminal():
                    return
                self.to_move = 1 - self._turn_owner()
                self.phase = PHASE_ACTION
                return

            if kind == "lose":
                p = op[1]
                if not self.hands[p]:
                    continue
                if len(self.hands[p]) == 1:
                    c = self.hands[p][0]
                    self.hands[p] = ()
                    self.revealed = tuple(sorted(self.revealed + (c,)))
                    continue
                self.phase = PHASE_DISCARD
                self.to_move = p
                return

            if kind == "redraw":
                p, card = op[1], op[2]
                if card not in self.hands[p] or not sum(self.deck):
                    continue
                if self.claim_memory:
                    # that card went back to the deck, so the claim proves nothing
                    # about the hand from here on; other claims still bind
                    self.claims[p] &= ~(1 << card)
                self._return(p, card)
                self.phase = PHASE_CHANCE
                self.chance = (REDRAW, p)
                return

            if kind == "resolve":
                actor, a = op[1], op[2]
                if not self.hands[actor]:
                    continue
                self._resolve(actor, a)
                continue

    def _turn_owner(self):
        """Whose turn just ended (turn parity)."""
        return (self.turn - 1) % 2

    def _resolve(self, actor, a):
        opp = 1 - actor
        if a == FOREIGN_AID:
            self.coins[actor] += 2
        elif a == TAX:
            self.coins[actor] += 3
        elif a == STEAL:
            t = min(2, self.coins[opp])
            self.coins[opp] -= t; self.coins[actor] += t
        elif a == ASSASSINATE:
            self.stack = self.stack + (("lose", opp),)
        elif a == EXCHANGE:
            if sum(self.deck) >= 1:
                self.pool = self.hands[actor]   # pool = your hand plus the cards you draw
                self.phase = PHASE_CHANCE
                self.chance = (EXDRAW, actor)


def new_game(max_turns=60, peek_memory=1, claim_memory=1):
    return State(max_turns=max_turns, peek_memory=peek_memory,
                 claim_memory=claim_memory)


def describe(state, player):
    """Human-readable rendering of one player's view. For the query tool."""
    opp = 1 - player
    hand = "+".join(CARD_NAMES[c] for c in state.hands[player]) or "-"
    rev = "+".join(CARD_NAMES[c] for c in state.revealed) or "none"
    def claimed(p):
        return "+".join(CARD_NAMES[c] for c in claim_cards(state.claims[p])) or "none"
    return (f"you {len(state.hands[player])} inf / {state.coins[player]}c ({hand})   "
            f"opp {len(state.hands[opp])} inf / {state.coins[opp]}c   face-up: {rev}   "
            f"claimed: you {claimed(player)} / opp {claimed(opp)}")
