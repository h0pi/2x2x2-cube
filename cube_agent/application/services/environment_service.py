from cube_agent.domain.cube_state import Cube2x2State
from cube_agent.domain.actions import ACTIONS


class EnvironmentService:
    """
    Manages a single Rubik's cube episode: current state, step counter,
    scrambling, action application, and reward calculation.

    Domain rules (reward shaping, max_steps) live here — not in any runner
    or UI layer.
    """

    def __init__(self, scramble_len: int = 5, max_steps: int = 30):
        self.scramble_len = scramble_len
        self.max_steps = max_steps
        self._state: Cube2x2State = Cube2x2State.solved()
        self._step_count: int = 0

    # ------------------------------------------------------------------ getters

    @property
    def step_count(self) -> int:
        return self._step_count

    def current_state_encoded(self) -> int:
        return self._state.encode()

    def current_cube_state(self) -> Cube2x2State:
        return self._state

    def is_solved(self) -> bool:
        return self._state.is_solved()

    # ------------------------------------------------------------------ episode control

    def reset(self, scramble_len: int = None) -> tuple[int, list[str]]:
        """
        Start a new episode: scramble the cube and reset the step counter.
        Returns (encoded_state, scramble_sequence).
        """
        self._step_count = 0
        n = scramble_len if scramble_len is not None else self.scramble_len
        self._state, seq = Cube2x2State.solved().scramble(n, ACTIONS)
        return self._state.encode(), seq

    def set_state(self, state: Cube2x2State) -> None:
        """Force a specific cube state (used for UI sync)."""
        self._state = state
        self._step_count = 0

    def prepare_for_solve(self) -> None:
        """Reset step counter so the agent can attempt a solve on the current cube state."""
        self._step_count = 0

    # ------------------------------------------------------------------ step

    def apply_action(self, action_str: str) -> tuple[int, float, bool]:
        """
        Apply one move, compute reward, and return (new_state_encoded, reward, done).

        Reward shaping (domain rule — never in Web/UI):
          -0.1  per step (encourages efficiency)
          ± heuristic delta × 2.0  (progress signal)
          +100  on solve
        """
        self._step_count += 1
        prev_h = self._state.heuristic()
        self._state = self._state.apply(action_str)
        new_h = self._state.heuristic()

        solved = self._state.is_solved()
        done = solved or (self._step_count >= self.max_steps)

        reward = -0.1
        reward += (prev_h - new_h) * 2.0
        if solved:
            reward += 100.0

        return self._state.encode(), reward, done
