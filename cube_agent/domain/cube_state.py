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
        # Never pick a move on the same face as the previous move —
        # prevents F followed by F' (or F followed by F) which wastes scramble depth.
        state = self
        seq = []
        last_face = None
        for _ in range(n):
            available = [m for m in actions if m[0] != last_face]
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


def _apply_cube_rotation(state: "Cube2x2State", pos_map: list, ori_map: list) -> "Cube2x2State":
    """Apply a whole-cube rotation via explicit position and orientation tables."""
    new_perm = [0] * 8
    new_ori  = [0] * 8
    for old_pos in range(8):
        new_pos = pos_map[old_pos]
        new_perm[new_pos] = state.perm[old_pos]
        new_ori[new_pos]  = ori_map[state.ori[old_pos]]
    return Cube2x2State(perm=tuple(new_perm), ori=tuple(new_ori))


def _generate_solved_states() -> frozenset:
    """
    BFS from the identity using two whole-cube rotation generators (x and y)
    to produce all 24 orientations of the solved cube.

    Rotation tables derived from the corner convention (0=UFR…7=DLF):

    x — CW from right (U→F, F→D, D→B, B→U):
      position map : 0→4, 1→0, 2→3, 3→7, 4→5, 5→1, 6→2, 7→6
      orientation  : 0→2 (U/D sticker was on y-axis → now on z-axis)
                     1→1 (R/L sticker stays on x-axis)
                     2→0 (F/B sticker was on z-axis → now on y-axis)

    y — CW from top (F→R, R→B, B→L, L→F):
      position map : 0→1, 1→2, 2→3, 3→0, 4→5, 5→6, 6→7, 7→4
      orientation  : 0→0 (U/D sticker stays on y-axis)
                     1→2 (R/L sticker was on x-axis → now on z-axis)
                     2→1 (F/B sticker was on z-axis → now on x-axis)
    """
    # pos_map[old_pos] = new_pos
    x_pos = [4, 0, 3, 7, 5, 1, 2, 6]
    x_ori = [2, 1, 0]

    y_pos = [1, 2, 3, 0, 5, 6, 7, 4]
    y_ori = [0, 2, 1]

    base = Cube2x2State(perm=tuple(range(8)), ori=(0,) * 8)

    seen: set = set()
    queue = [base]
    while queue:
        s = queue.pop()
        if s in seen:
            continue
        seen.add(s)
        for pos_map, ori_map in ((x_pos, x_ori), (y_pos, y_ori)):
            ns = _apply_cube_rotation(s, pos_map, ori_map)
            if ns not in seen:
                queue.append(ns)
    return frozenset(seen)


# Precomputed once at import time — all 24 orientations of a solved 2×2 cube
_SOLVED_STATES = _generate_solved_states()
