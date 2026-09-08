"""Kuhn poker -- a three-card game whose equilibrium is known on paper.

Used to prove the solver is correct before pointing it at Coup. The game value
to player 0 at equilibrium is exactly -1/18 = -0.05555..., and NashConv must go
to zero. If the solver cannot reproduce that, nothing it says about Coup counts.
"""

JACK, QUEEN, KING = 0, 1, 2
GAME_VALUE_P0 = -1.0 / 18.0


class KuhnState:
    __slots__ = ("cards", "history", "dealt")

    def __init__(self):
        self.cards = [None, None]
        self.history = ""
        self.dealt = 0

    def clone(self):
        s = KuhnState.__new__(KuhnState)
        s.cards = list(self.cards); s.history = self.history; s.dealt = self.dealt
        return s

    def is_chance(self):
        return self.dealt < 2

    def chance_outcomes(self):
        left = [c for c in (JACK, QUEEN, KING) if c not in self.cards[:self.dealt]]
        return [(c, 1.0 / len(left)) for c in left]

    def apply_chance(self, card):
        s = self.clone()
        s.cards[s.dealt] = card
        s.dealt += 1
        return s

    def is_terminal(self):
        if self.dealt < 2:
            return False
        h = self.history
        return h in ("pp", "bp", "bb", "pbp", "pbb")

    def utility(self, player):
        h = self.history
        me, opp = self.cards[player], self.cards[1 - player]
        win = 1 if me > opp else -1
        if h == "pp":
            return float(win)
        if h == "bp":                       # p0 bet, p1 folded
            return 1.0 if player == 0 else -1.0
        if h == "pbp":                      # p1 bet, p0 folded
            return -1.0 if player == 0 else 1.0
        return 2.0 * win                    # bb / pbb : showdown for 2

    def current_player(self):
        return len(self.history) % 2

    def legal(self):
        return ("p", "b")                   # pass/check-fold, bet/call

    def infoset_key(self, player):
        return (self.cards[player], self.history)

    def apply(self, action):
        s = self.clone()
        s.history += action
        return s


def new_kuhn():
    return KuhnState()
