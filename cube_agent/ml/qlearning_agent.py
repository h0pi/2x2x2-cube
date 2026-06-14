import random
import threading
import numpy as np

from cube_agent.ml.i_cube_policy import ICubePolicy
from cube_agent.domain.actions import ACTIONS

# 8! permutations × 3^7 valid orientations (8th is always determined by sum≡0 mod 3)
_STATE_SPACE = 40320 * 2187   # 88,179,840
_N_ACTIONS   = len(ACTIONS)   # 12


class QLearningAgent(ICubePolicy):
    """
    Tabular Q-learning backed by a numpy float32 array instead of a Python dict.

    Memory comparison (65 M visited entries):
      defaultdict + tuple keys : ~200 bytes/entry  →  ~13 GB physical RAM
      numpy float32 array      :   4 bytes/entry   →  ~260 MB physical RAM

    The array is 88 M × 12 entries (~4.2 GB virtual address space) but uses
    demand-paged zero pages on Windows and Linux — physical RAM is only
    allocated for pages that are actually written during training.

    Thread-safe: background training thread and UI solving thread share this
    object via an RLock.
    """

    def __init__(self, alpha: float = 0.2, gamma: float = 0.95,
                 eps: float = 1.0, eps_min: float = 0.05,
                 eps_decay: float = 0.995):
        self._alpha     = alpha
        self._gamma     = gamma
        self._eps       = eps
        self._eps_min   = eps_min
        self._eps_decay = eps_decay
        # Demand-paged: physical RAM grows only as training visits new states
        self._Q    = np.zeros((_STATE_SPACE, _N_ACTIONS), dtype=np.float32)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ props

    @property
    def eps(self) -> float:
        return self._eps

    # ------------------------------------------------------------------ policy

    def choose_action_greedy(self, state_encoded: int) -> int:
        """Pure greedy — used by the solving agent (no exploration)."""
        with self._lock:
            row   = self._Q[state_encoded]
            max_q = float(np.max(row))
            # Break ties randomly so the agent doesn't get stuck in a loop
            candidates = np.flatnonzero(row == max_q)
            return int(np.random.choice(candidates))

    def choose_action(self, state_encoded: int) -> int:
        """Epsilon-greedy — used by the training agent (with exploration)."""
        if random.random() < self._eps:
            return random.randrange(_N_ACTIONS)
        return self.choose_action_greedy(state_encoded)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        with self._lock:
            max_next = 0.0 if done else float(np.max(self._Q[next_state]))
            old    = float(self._Q[state, action])
            target = reward + self._gamma * max_next
            self._Q[state, action] = np.float32(old + self._alpha * (target - old))

    def decay_eps(self) -> None:
        self._eps = max(self._eps_min, self._eps * self._eps_decay)

    def get_action_values(self, state_encoded: int) -> list:
        with self._lock:
            return self._Q[state_encoded].tolist()

    def set_eps(self, value: float) -> None:
        self._eps = max(self._eps_min, min(1.0, value))

    # ------------------------------------------------------------------ persistence

    def get_qtable(self) -> dict:
        """Return a compact sparse snapshot for pickling (only non-zero entries)."""
        with self._lock:
            rows, cols = np.nonzero(self._Q)
            return {
                'fmt':    'sparse_v2',
                'rows':   rows.astype(np.int32),
                'cols':   cols.astype(np.uint8),
                'values': self._Q[rows, cols].copy(),
                'eps':    self._eps,
            }

    def set_qtable(self, data: dict) -> None:
        with self._lock:
            if not isinstance(data, dict):
                return
            if data.get('fmt') == 'sparse_v2':
                self._Q[data['rows'], data['cols']] = data['values']
                self._eps = max(self._eps_min, float(data.get('eps', 0.3)))
            else:
                # Old defaultdict format — encoding has changed, cannot reuse
                print("[QLearningAgent] Old Q-table format detected — starting fresh.")
