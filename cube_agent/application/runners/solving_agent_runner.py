from dataclasses import dataclass, field
from typing import Optional

from agents_core.base_agent import SoftwareAgent
from cube_agent.domain.actions import IDX_TO_ACTION, ACTIONS
from cube_agent.domain.results import SolvingTickResult
from cube_agent.application.services.environment_service import EnvironmentService
from cube_agent.ml.i_cube_policy import ICubePolicy

_N_ACTIONS = len(ACTIONS)


@dataclass
class _CubePercept:
    """What the solving agent observes at the start of each tick."""
    state_encoded: int
    step_count: int


class SolvingAgentRunner(SoftwareAgent[_CubePercept, int, SolvingTickResult, None]):
    """
    Agent that solves a scrambled cube one move at a time.

    Each step() is one Sense->Think->Act cycle:
      Sense : read current cube state from the shared EnvironmentService
      Think : pick the highest-Q action that does NOT lead to a already-visited
              state (loop prevention); fall back to highest-Q if all next states
              have been visited
      Act   : apply the action, update environment state
      (no Learn phase — this agent does not update the Q-table)

    Returns None when the cube is already solved or max_steps is reached.
    """

    def __init__(self, env_service: EnvironmentService, policy: ICubePolicy):
        self._env = env_service
        self._policy = policy
        self._visited: set[int] = set()

    def reset_visited(self) -> None:
        """Call this when starting a new solve attempt."""
        self._visited.clear()

    def step(self) -> Optional[SolvingTickResult]:
        # SENSE
        percept = self._sense()
        if percept is None:
            return None  # no work: already solved or out of steps

        # THINK
        action_idx = self._think(percept)

        # ACT
        return self._act(percept, action_idx)

    # ------------------------------------------------------------------ internals

    def _sense(self) -> Optional[_CubePercept]:
        if self._env.is_solved() or self._env.step_count >= self._env.max_steps:
            return None
        return _CubePercept(
            state_encoded=self._env.current_state_encoded(),
            step_count=self._env.step_count,
        )

    def _think(self, percept: _CubePercept) -> int:
        self._visited.add(percept.state_encoded)
        q_vals = self._policy.get_action_values(percept.state_encoded)

        # Sort actions best-first; pick the first whose next state is unvisited
        sorted_actions = sorted(range(_N_ACTIONS), key=lambda a: -q_vals[a])
        cube_state = self._env.current_cube_state()
        for action_idx in sorted_actions:
            next_enc = cube_state.apply(IDX_TO_ACTION[action_idx]).encode()
            if next_enc not in self._visited:
                return action_idx

        # Every next state has been visited — just take the best Q action
        return sorted_actions[0]

    def _act(self, percept: _CubePercept, action_idx: int) -> SolvingTickResult:
        action_str = IDX_TO_ACTION[action_idx]
        _, _, done = self._env.apply_action(action_str)
        return SolvingTickResult(
            action_str=action_str,
            action_idx=action_idx,
            is_solved=self._env.is_solved(),
            step_count=self._env.step_count,
            done=done,
        )
