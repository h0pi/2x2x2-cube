import random
from moves2x2 import ACTIONS

def generate_random_movements(n):
    return [random.choice(ACTIONS) for _ in range(n)]