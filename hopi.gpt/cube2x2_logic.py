# cube2x2_logic.py
from dataclasses import dataclass
from typing import Tuple
import random

# Corner index convention (bitno je da bude konzistentno):
# 0=UFR, 1=URB, 2=UBL, 3=ULF, 4=DFR, 5=DRB, 6=DBL, 7=DLF

# Move tables: za svaku face rotaciju kaže kako se permutacija cornera mijenja
# (ovo je CW rotacija gledano u tu stranu face-a).
MOVE_CYCLES = {
    "U":  (0, 1, 2, 3),
    "D":  (4, 7, 6, 5),
    "F":  (0, 3, 7, 4),
    "B":  (1, 5, 6, 2),
    "R":  (0, 4, 5, 1),
    "L":  (2, 6, 7, 3),
}

# Corner orientation change for 2x2:
# U/D ne mijenja orijentacije cornera.
# F/B/R/L mijenjaju orijentacije cornera (twist).
# Ovo je klasičan model: na face turnu, 4 cornera dobiju +1/+2 mod 3 zavisno od pozicije.
ORI_DELTA = {
    "U":  (0, 0, 0, 0),
    "D":  (0, 0, 0, 0),
    "F":  (1, 2, 1, 2),  # corners in cycle order
    "B":  (1, 2, 1, 2),
    "R":  (1, 2, 1, 2),
    "L":  (1, 2, 1, 2),
}

def cycle4(arr, a, b, c, d):
    arr[a], arr[b], arr[c], arr[d] = arr[d], arr[a], arr[b], arr[c]

@dataclass(frozen=True)
class Cube2x2State:
    perm: Tuple[int, ...]  # length 8
    ori: Tuple[int, ...]   # length 8, each 0..2

    @staticmethod
    def solved():
        return Cube2x2State(perm=tuple(range(8)), ori=(0,)*8)

    def is_solved(self) -> bool:
        return self.perm == tuple(range(8)) and self.ori == (0,)*8

    def apply(self, move: str) -> "Cube2x2State":
        # move can be "U", "U'", etc.
        base = move[0]
        times = 1 if len(move) == 1 else 3  # prime = 3 CW turns

        perm = list(self.perm)
        ori = list(self.ori)

        for _ in range(times):
            a, b, c, d = MOVE_CYCLES[base]

            # perm cycle
            cycle4(perm, a, b, c, d)

            # ori update for affected corners (only for some moves)
            deltas = ORI_DELTA[base]
            if deltas != (0, 0, 0, 0):
                # after cycling corners, orientation of pieces moved into positions a,b,c,d changes
                # Apply delta to the *piece now at position* a,b,c,d
                for pos, delta in zip((a, b, c, d), deltas):
                    ori[pos] = (ori[pos] + delta) % 3

        return Cube2x2State(perm=tuple(perm), ori=tuple(ori))

    def scramble(self, n: int, actions):
        s = self
        seq = []
        for _ in range(n):
            m = random.choice(actions)
            s = s.apply(m)
            seq.append(m)
        return s, seq

    def encode(self) -> int:
        # Encode perm (factoradic) + ori (base-3)
        # perm -> [0..40319], ori -> [0..3^8-1]
        perm_code = self._perm_to_int(self.perm)
        ori_code = 0
        for i in range(8):
            ori_code = ori_code * 3 + self.ori[i]
        return perm_code * (3**8) + ori_code

    @staticmethod
    def _perm_to_int(p: Tuple[int, ...]) -> int:
        # factoradic ranking
        items = list(p)
        rank = 0
        fact = 1
        for i in range(7, 0, -1):
            fact *= (8 - i)
        # simpler:
        import math
        rank = 0
        elems = list(p)
        for i in range(8):
            smaller = sum(1 for j in elems[i+1:] if j < elems[i])
            rank += smaller * math.factorial(7 - i)
        return rank

    def heuristic(self) -> int:
        # jednostavna heuristika: broj pogrešnih cornera + pogrešnih orijentacija
        wrong_pos = sum(1 for i, v in enumerate(self.perm) if v != i)
        wrong_ori = sum(1 for o in self.ori if o != 0)
        return wrong_pos + wrong_ori
