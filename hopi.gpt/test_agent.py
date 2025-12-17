from env2x2 import Rubik2x2Env
from qlearning_agent import QLearningAgent
from moves2x2 import ACTIONS
import pickle
from collections import defaultdict

# učitaj env
env = Rubik2x2Env(scramble_len=5, max_steps=30)

# učitaj agenta
agent = QLearningAgent(ACTIONS)
agent.eps = 0.0   # BEZ RANDOMA

# učitaj Q-tabelu
with open("qtable_2x2.pkl", "rb") as f:
    loaded_Q = pickle.load(f)

agent.Q = defaultdict(float, loaded_Q)

# testiraj više epizoda
SOLVED = 0
TEST_EPISODES = 100

for i in range(TEST_EPISODES):
    state, _ = env.reset()
    steps = 0
    done = False

    while not done:
        a = agent.choose_action(state)
        move = ACTIONS[a]

        state, reward, done = env.step(move)
        steps += 1

        if env.is_solved():
            SOLVED += 1
            break

    print(f"[TEST {i+1}] steps={steps} solved={env.is_solved()}")

print(f"\nSolved {SOLVED}/{TEST_EPISODES} episodes")
