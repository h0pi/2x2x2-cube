from dataclasses import dataclass
from typing import Optional

from agents_core.base_agent import SoftwareAgent
from cube_agent.domain.actions import IDX_TO_ACTION
from cube_agent.domain.results import SolvingTickResult
from cube_agent.application.services.environment_service import EnvironmentService
from cube_agent.ml.i_cube_policy import ICubePolicy


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
      Think : pick the best action via a pure-greedy policy lookup (epsilon=0)
      Act   : apply the action, update environment state
      (no Learn phase — this agent does not update the Q-table)

    Returns None when the cube is already solved or max_steps is reached
    (no-work case — host should stop calling step() and wait for reset).

    The runner operates on the *shared* EnvironmentService, keeping the
    UI visual cube and the logical state in sync.
    """

    def __init__(self, env_service: EnvironmentService, policy: ICubePolicy):
        self._env = env_service
        self._policy = policy

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
        return self._policy.choose_action_greedy(percept.state_encoded)

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
