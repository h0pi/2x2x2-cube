from abc import ABC, abstractmethod


class ICubePolicy(ABC):
    """
    Abstract interface for cube-solving policies.
    Implementations must be thread-safe — training runs in a background
    thread while solving reads from the same policy on the UI thread.
    """

    @abstractmethod
    def choose_action_greedy(self, state_encoded: int) -> int:
        """Pure greedy: always pick the action with the highest Q-value."""
        ...

    @abstractmethod
    def choose_action(self, state_encoded: int) -> int:
        """Epsilon-greedy: explore randomly or exploit greedily."""
        ...

    @abstractmethod
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        """Update the policy given a (s, a, r, s', done) transition."""
        ...

    @abstractmethod
    def decay_eps(self) -> None:
        """Reduce exploration rate after each completed episode."""
        ...

    @property
    @abstractmethod
    def eps(self) -> float:
        """Current exploration rate (epsilon)."""
        ...

    @abstractmethod
    def get_qtable(self) -> dict:
        """Return a snapshot of the Q-table for persistence."""
        ...

    @abstractmethod
    def set_qtable(self, data: dict) -> None:
        """Replace the Q-table from a loaded snapshot."""
        ...
