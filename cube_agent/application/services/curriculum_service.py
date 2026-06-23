from cube_agent.application.runners.training_agent_runner import TrainingAgentRunner
from cube_agent.ml.i_cube_policy import ICubePolicy

# ------------------------------------------------------------------ curriculum constants

MAX_DEPTH             = 11     # God's number for 2×2×2
_PROMOTE_RATE         = 0.80   # window solve rate needed to advance one depth
_WINDOW_SIZE          = 500    # episodes per evaluation window
_MAX_STAGNANT_WINDOWS = 5      # consecutive non-promoting windows before forcing exploration
_EPS_ON_PROMOTE       = 0.50   # epsilon injected when depth advances
_EPS_ON_STAGNANT      = 0.30   # epsilon injected when stuck too long


def max_steps_for_depth(depth: int) -> int:
    """
    Scale the episode step limit with scramble depth.
    At depth 1 (1-move scramble), 30 steps is wasteful — the agent wanders for
    29 steps generating noisy updates after the trivial solve.
    Formula: max(depth × 3, 20) keeps a reasonable floor and grows with depth.
      depth  1 →  20   depth  7 →  21
      depth  5 →  20   depth 11 →  33
    """
    return max(depth * 3, 20)


class CurriculumService:
    """
    Owns curriculum-learning decisions for training: when to promote scramble
    depth, when to force exploration on a plateau, and how the step limit
    scales with depth.

    This is agent/training policy, not UI — it must work the same whether or
    not a GUI is attached, so it lives in the application layer and only
    depends on the runner + policy interfaces, never on raylib or threading.

    Usage: call record_episode() once per finished episode (episode_done is
    True); it updates internal stats and applies any promotion/exploration
    boost to the runner and policy.
    """

    def __init__(self, runner: TrainingAgentRunner, policy: ICubePolicy,
                 start_depth: int = 1):
        self._runner = runner
        self._policy = policy

        self._episodes:     int = 0
        self._solved_count: int = 0

        self._current_depth:    int = start_depth
        self._window_episodes:  int = 0
        self._window_solved:    int = 0
        self._stagnant_windows: int = 0   # consecutive non-promoting windows

    # ------------------------------------------------------------------ stats

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def solved_count(self) -> int:
        return self._solved_count

    @property
    def current_depth(self) -> int:
        return self._current_depth

    @property
    def window_solve_rate(self) -> float:
        """Live solve rate of the current evaluation window (resets every
        _WINDOW_SIZE episodes). More meaningful than lifetime rate, which is
        diluted by easy early-depth episodes."""
        if self._window_episodes == 0:
            return 0.0
        return self._window_solved / self._window_episodes

    def solve_rate(self) -> float:
        """Lifetime solve rate across all depths (kept for compatibility)."""
        if self._episodes == 0:
            return 0.0
        return self._solved_count / self._episodes

    # ------------------------------------------------------------------ evaluation

    def record_episode(self, episode_count: int, solved: bool) -> None:
        """
        Call once per finished training episode. Updates stats and, once a
        full evaluation window has elapsed, decides whether to promote the
        scramble depth or boost exploration on a plateau — applying the
        decision directly to the runner/policy.
        """
        self._episodes = episode_count
        if solved:
            self._solved_count += 1

        self._window_episodes += 1
        if solved:
            self._window_solved += 1

        if self._window_episodes < _WINDOW_SIZE:
            return

        window_rate = self._window_solved / self._window_episodes
        self._window_episodes = 0
        self._window_solved   = 0

        if window_rate >= _PROMOTE_RATE and self._current_depth < MAX_DEPTH:
            self._promote(window_rate)
        else:
            self._stagnant_windows += 1
            if self._stagnant_windows >= _MAX_STAGNANT_WINDOWS:
                self._stagnant_windows = 0
                self._boost_exploration()

    # ------------------------------------------------------------------ internals

    def _promote(self, window_rate: float) -> None:
        old_depth = self._current_depth
        self._current_depth += 1
        self._stagnant_windows = 0

        self._runner.set_scramble_len(self._current_depth)
        self._runner.set_max_steps(max_steps_for_depth(self._current_depth))
        self._policy.set_eps(_EPS_ON_PROMOTE)

        print(
            f"[Curriculum] depth {old_depth} → {self._current_depth}  "
            f"(window rate: {window_rate:.1%})  "
            f"eps reset to {_EPS_ON_PROMOTE}"
        )

    def _boost_exploration(self) -> None:
        self._policy.set_eps(_EPS_ON_STAGNANT)
        print(
            f"[Curriculum] Plateau at depth {self._current_depth} "
            f"after {_MAX_STAGNANT_WINDOWS} windows — "
            f"boosting eps to {_EPS_ON_STAGNANT}"
        )
