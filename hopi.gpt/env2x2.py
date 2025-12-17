from cube2x2_logic import Cube2x2State
from moves2x2 import ACTIONS

# helper za inverse potez
INVERSE = {
    "U": "U'", "U'": "U",
    "D": "D'", "D'": "D",
    "L": "L'", "L'": "L",
    "R": "R'", "R'": "R",
    "F": "F'", "F'": "F",
    "B": "B'", "B'": "B",
}

def inverse_of(a):
    if a is None:
        return None
    return INVERSE[a]


class Rubik2x2Env:
    def __init__(self, scramble_len=5, max_steps=30):
        self.scramble_len = scramble_len
        self.max_steps = max_steps
        self.state = Cube2x2State.solved()
        self.steps = 0
        self.last_action = None

    def reset(self, scramble_len=None):
        self.steps = 0
        self.last_action = None

        s = Cube2x2State.solved()
        n = scramble_len if scramble_len is not None else self.scramble_len
        s, seq = s.scramble(n, ACTIONS)

        self.state = s
        return self.state.encode(), seq

    def step(self, action_str: str):
        self.steps += 1

        prev_h = self.state.heuristic()

        # primijeni potez TAČNO JEDNOM
        self.state = self.state.apply(action_str)

        new_h = self.state.heuristic()
        solved = self.state.is_solved()
        done = solved or (self.steps >= self.max_steps)

        # osnovna kazna po potezu
        reward = -0.1

        # kazna za undo potez
        #if action_str == inverse_of(self.last_action):
            #reward -= 1.0

        # heuristički signal
        reward += (prev_h - new_h) * 2.0

        if solved:
            reward += 100.0

        # zapamti zadnji potez
        #self.last_action = action_str

        return self.state.encode(), reward, done

    def is_solved(self):
        return self.state.is_solved()
