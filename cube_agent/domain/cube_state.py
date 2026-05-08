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
    "B": (1, 2, 1, 2),
    "R": (1, 2, 1, 2),
    "L": (1, 2, 1, 2),
}


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
        state = self
        seq = []
        for _ in range(n):
            m = random.choice(actions)
            state = state.apply(m)
            seq.append(m)
        return state, seq

    def encode(self) -> int:
        perm_code = _perm_to_int(self.perm)
        ori_code = 0
        for o in self.ori:
            ori_code = ori_code * 3 + o
        return perm_code * (3 ** 8) + ori_code

    def heuristic(self) -> int:
        wrong_pos = sum(1 for i, v in enumerate(self.perm) if v != i)
        wrong_ori = sum(1 for o in self.ori if o != 0)
        return wrong_pos + wrong_ori


def _perm_to_int(p: Tuple[int, ...]) -> int:
    rank = 0
    elems = list(p)
    for i in range(8):
        smaller = sum(1 for j in elems[i + 1:] if j < elems[i])
        rank += smaller * math.factorial(7 - i)
    return rank
