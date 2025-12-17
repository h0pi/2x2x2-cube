from env2x2 import Rubik2x2Env
from qlearning_agent import QLearningAgent
from moves2x2 import ACTIONS
import pickle
from collections import defaultdict
import os


env = Rubik2x2Env(scramble_len=5, max_steps=30)
agent = QLearningAgent(ACTIONS)

if os.path.exists("qtable_2x2.pkl"):
    with open("qtable_2x2.pkl", "rb") as f:
        loaded_Q = pickle.load(f)
    agent.Q = defaultdict(float, loaded_Q)
    agent.eps = max(agent.eps, 0.3)
    print("Loaded existing Q-table.")
else:
    print("Starting training from scratch.")

state, _ = env.reset()
episode = 0

while episode<1000000:   # ili while True
    a = agent.choose_action(state)
    move = ACTIONS[a]

    next_state, reward, done = env.step(move)
    agent.update(state, a, reward, next_state, done)

    state = next_state

    if done:
        episode += 1
        agent.decay_eps()
        state, _ = env.reset()

        if episode % 100 == 0:
            print(f"[TRAIN] EP {episode} eps={agent.eps:.3f}")

with open("qtable_2x2.pkl", "wb") as f:
    pickle.dump(dict(agent.Q), f)

print("Training finished, Q-table saved.")

