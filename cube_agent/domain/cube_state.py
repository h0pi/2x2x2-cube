from dataclasses import dataclass
from typing import Tuple
import random
import math

# Corner index convention:
# 0=UFR, 1=URB, 2=UBL, 3=ULF, 4=DFR, 5=DRB, 6=DBL, 7=DLF

MOVE_CYCLES = {
    "U": (0, 1, 2, 3),
    "D": (4, 7, 6, 5),
    "F": (0, 3, 7, 4),
    "B": (1, 5, 6, 2),
    "R": (0, 4, 5, 1),
    "L": (2, 6, 7, 3),
}

ORI_DELTA = {
    "U": (0, 0, 0, 0),
    "D": (0, 0, 0, 0),
    "F": (1, 2, 1, 2),
    "B": (2, 1, 2, 1),
    "R": (2, 1, 2, 1),
    "L": (2, 1, 2, 1),
}

# Faces that cancel each other out on a 2×2 (they share no corners and commute,
# so U then D is equivalent to a whole-cube rotation — wastes a scramble move).
_OPPOSITE_FACE = {'U': 'D', 'D': 'U', 'R': 'L', 'L': 'R', 'F': 'B', 'B': 'F'}


def _cycle4(arr, a, b, c, d):
    arr[a], arr[b], arr[c], arr[d] = arr[d], arr[a], arr[b], arr[c]


@dataclass(frozen=True)
class Cube2x2State:
    perm: Tuple[int, ...]  # length 8 — corner permutation
    ori: Tuple[int, ...]   # length 8 — corner orientation (0..2)

    @staticmethod
    def solved() -> "Cube2x2State":
        return Cube2x2State(perm=tuple(range(8)), ori=(0,) * 8)

    def is_solved(self) -> bool:
        return self.perm == tuple(range(8)) and self.ori == (0,) * 8

    def apply(self, move: str) -> "Cube2x2State":
        base = move[0]
        times = 1 if len(move) == 1 else 3  # prime = 3 CW turns

        perm = list(self.perm)
        ori = list(self.ori)

        for _ in range(times):
            a, b, c, d = MOVE_CYCLES[base]
            _cycle4(perm, a, b, c, d)
            _cycle4(ori, a, b, c, d)
            deltas = ORI_DELTA[base]
            if deltas != (0, 0, 0, 0):
                for pos, delta in zip((a, b, c, d), deltas):
                    ori[pos] = (ori[pos] + delta) % 3

        return Cube2x2State(perm=tuple(perm), ori=tuple(ori))

    def scramble(self, n: int, actions: list) -> tuple["Cube2x2State", list]:
        """
        Generate a random scramble of length n.

        Filters out two classes of useless move sequences:
          - Same face back-to-back (F then F or F then F') — cancels or duplicates.
          - Opposite face back-to-back (U then D) — they commute on a 2×2,
            so their combined effect is just a whole-cube rotation, wasting depth.
        Both filters together ensure every scramble move adds genuine complexity.
        With 6 faces, avoiding same+opposite still leaves 4 faces × 2 moves = 8
        choices at every step, so random.choice is always well-defined.
        """
        state = self
        seq = []
        last_face = None
        for _ in range(n):
            avoid = {last_face, _OPPOSITE_FACE.get(last_face)}
            available = [m for m in actions if m[0] not in avoid]
            m = random.choice(available)
            state = state.apply(m)
            seq.append(m)
            last_face = m[0]
        return state, seq

    def encode(self) -> int:
        # The 8th orientation is always determined (sum of all ori ≡ 0 mod 3),
        # so encoding only the first 7 is sufficient:
        # 8! × 3^7 = 88,179,840  vs  8! × 3^8 = 264,539,520  (old)
        perm_code = _perm_to_int(self.perm)
        ori_code = 0
        for o in self.ori[:7]:
            ori_code = ori_code * 3 + o
        return perm_code * 2187 + ori_code

    def heuristic(self) -> int:
        """
        Count of corners that are not fully solved (wrong position OR wrong orientation).

        Range: 0 (solved) to 8 (all corners wrong).

        Replaces the old wrong_pos + wrong_ori which double-counted corners that
        had BOTH a wrong position and wrong orientation (range 0-16), producing
        an inflated signal that could mislead the reward shaping at higher depths.
        Each corner is now counted at most once.
        """
        return sum(
            1 for i in range(8)
            if self.perm[i] != i or self.ori[i] != 0
        )


def _perm_to_int(p: Tuple[int, ...]) -> int:
    rank = 0
    elems = list(p)
    for i in range(8):
        smaller = sum(1 for j in elems[i + 1:] if j < elems[i])
        rank += smaller * math.factorial(7 - i)
    return rank
