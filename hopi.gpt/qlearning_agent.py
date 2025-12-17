# qlearning_agent.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=0.995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = defaultdict(float)  # key: (state, action_idx)

    def choose_action(self, state):
        best_q = float("-inf")
        best_actions = []

        for a in range(len(self.actions)):
            q = self.Q[(state, a)]
            if q > best_q:
                best_q = q
                best_actions = [a]
            elif q == best_q:
                best_actions.append(a)

        return random.choice(best_actions)

        

    def update(self, s, a, r, s2, done):
        # max_a' Q(s2,a')
        max_next = 0.0
        if not done:
            max_next = max(self.Q[(s2, a2)] for a2 in range(len(self.actions)))

        old = self.Q[(s, a)]
        target = r + self.gamma * max_next
        self.Q[(s, a)] = old + self.alpha * (target - old)

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
