import numpy as np

# moves2x2.py
ACTIONS = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}
