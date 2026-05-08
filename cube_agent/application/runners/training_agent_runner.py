from dataclasses import dataclass

from agents_core.base_agent import SoftwareAgent
from cube_agent.domain.actions import IDX_TO_ACTION
from cube_agent.domain.results import TrainingTickResult
from cube_agent.application.services.environment_service import EnvironmentService
from cube_agent.ml.i_cube_policy import ICubePolicy


@dataclass
class _TrainingPercept:
    """What the training agent observes at the start of each tick."""
    state_encoded: int
    epsilon: float


class TrainingAgentRunner(SoftwareAgent[_TrainingPercept, int, TrainingTickResult, tuple]):
    """
    Agent that trains the Q-learning policy by running episodes headlessly.

    Each step() is one Sense->Think->Act->Learn cycle:
      Sense  : read current cube state + current epsilon
      Think  : choose action with epsilon-greedy exploration
      Act    : apply action, collect reward, advance environment
      Learn  : Bellman update of Q(s,a); decay epsilon + reset if episode done

    Always returns a TrainingTickResult (never None) — there is always a
    training step to execute.  The runner manages its own private
    EnvironmentService, independent of the UI environment.
    """

    def __init__(self, policy: ICubePolicy,
                 scramble_len: int = 5, max_steps: int = 30):
        self._policy = policy
        self._env = EnvironmentService(scramble_len=scramble_len,
                                       max_steps=max_steps)
        self._episode_count: int = 0
        self._env.reset()  # start first episode immediately

    def step(self) -> TrainingTickResult:
        # SENSE
        percept = self._sense()

        # THINK (epsilon-greedy — exploration is essential during training)
        action_idx = self._think(percept)
        action_str = IDX_TO_ACTION[action_idx]

        # ACT
        prev_state_enc = percept.state_encoded
        new_state_enc, reward, done = self._env.apply_action(action_str)

        # LEARN
        self._learn(prev_state_enc, action_idx, reward, new_state_enc, done)

        if done:
            solved = self._env.is_solved()
            self._policy.decay_eps()
            self._env.reset()
            self._episode_count += 1
            return TrainingTickResult(
                episode_done=True,
                solved=solved,
                epsilon=self._policy.eps,
                episode_count=self._episode_count,
            )

        return TrainingTickResult(
            episode_done=False,
            solved=False,
            epsilon=self._policy.eps,
            episode_count=self._episode_count,
        )

    # ------------------------------------------------------------------ internals

    def _sense(self) -> _TrainingPercept:
        return _TrainingPercept(
            state_encoded=self._env.current_state_encoded(),
            epsilon=self._policy.eps,
        )

    def _think(self, percept: _TrainingPercept) -> int:
        return self._policy.choose_action(percept.state_encoded)

    def _learn(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        self._policy.update(state, action, reward, next_state, done)
