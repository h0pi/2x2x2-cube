import random
import threading
from collections import defaultdict

from cube_agent.ml.i_cube_policy import ICubePolicy
from cube_agent.domain.actions import ACTIONS


class QLearningAgent(ICubePolicy):
    """
    Tabular Q-learning policy.

    Thread-safe: a background training thread and the UI solving thread
    both access the Q-table, protected by an RLock.
    Fixed bug from original: choose_action now properly uses epsilon-greedy
    exploration during training (original code was always greedy).
    """

    def __init__(self, alpha: float = 0.2, gamma: float = 0.95,
                 eps: float = 1.0, eps_min: float = 0.05,
                 eps_decay: float = 0.995):
        self._alpha = alpha
        self._gamma = gamma
        self._eps = eps
        self._eps_min = eps_min
        self._eps_decay = eps_decay
        self._Q: dict = defaultdict(float)
        self._lock = threading.RLock()
        self._n_actions = len(ACTIONS)

    # ------------------------------------------------------------------ props

    @property
    def eps(self) -> float:
        return self._eps

    # ------------------------------------------------------------------ policy

    def choose_action_greedy(self, state_encoded: int) -> int:
        """Pure greedy — used by the solving agent (no exploration)."""
        with self._lock:
            best_q = float('-inf')
            best = []
            for a in range(self._n_actions):
                q = self._Q[(state_encoded, a)]
                if q > best_q:
                    best_q = q
                    best = [a]
                elif q == best_q:
                    best.append(a)
            return random.choice(best)

    def choose_action(self, state_encoded: int) -> int:
        """Epsilon-greedy — used by the training agent (with exploration)."""
        if random.random() < self._eps:
            return random.randrange(self._n_actions)
        return self.choose_action_greedy(state_encoded)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        with self._lock:
            max_next = 0.0
            if not done:
                max_next = max(
                    self._Q[(next_state, a)] for a in range(self._n_actions)
                )
            old = self._Q[(state, action)]
            target = reward + self._gamma * max_next
            self._Q[(state, action)] = old + self._alpha * (target - old)

    def decay_eps(self) -> None:
        self._eps = max(self._eps_min, self._eps * self._eps_decay)

    # ------------------------------------------------------------------ persistence

    def get_qtable(self) -> dict:
        with self._lock:
            return dict(self._Q)

    def set_qtable(self, data: dict) -> None:
        with self._lock:
            self._Q = defaultdict(float, data)
            # Resume training from a moderate exploration rate
            self._eps = max(self._eps_min, 0.3)
