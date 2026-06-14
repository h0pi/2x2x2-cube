import threading

from cube_agent.application.runners.training_agent_runner import TrainingAgentRunner
from cube_agent.application.services.training_service import TrainingService
from cube_agent.ml.i_cube_policy import ICubePolicy

# ------------------------------------------------------------------ curriculum constants

_MAX_DEPTH            = 11     # God's number for 2×2×2
_PROMOTE_RATE         = 0.80   # window solve rate needed to advance one depth
_WINDOW_SIZE          = 500    # episodes per evaluation window
_MAX_STAGNANT_WINDOWS = 5      # consecutive non-promoting windows before forcing exploration
_EPS_ON_PROMOTE       = 0.50   # epsilon injected when depth advances
_EPS_ON_STAGNANT      = 0.30   # epsilon injected when stuck too long


def _max_steps_for_depth(depth: int) -> int:
    """
    Scale the episode step limit with scramble depth.
    At depth 1 (1-move scramble), 30 steps is wasteful — the agent wanders for
    29 steps generating noisy updates after the trivial solve.
    Formula: max(depth × 3, 20) keeps a reasonable floor and grows with depth.
      depth  1 →  20   depth  7 →  21
      depth  5 →  20   depth 11 →  33
    """
    return max(depth * 3, 20)


class TrainingWorker:
    """
    Background thread that drives the TrainingAgentRunner in a tight loop.

    Manages:
      - threading lifecycle (start / stop)
      - curriculum learning:
          * starts at scramble depth 1, promotes to depth 11 as solve rate improves
          * boosts epsilon (_EPS_ON_PROMOTE) whenever the depth increases so the
            agent can explore the harder state space rather than being trapped in
            near-min epsilon exploitation
          * detects plateaus: if the agent fails to hit the promotion threshold for
            _MAX_STAGNANT_WINDOWS consecutive windows, epsilon is boosted
            (_EPS_ON_STAGNANT) to force exploration
          * scales max_steps with depth (_max_steps_for_depth)
      - periodic Q-table saves
      - per-window stats (window_solve_rate) for live HUD feedback — more useful
        than the lifetime solve_rate which is diluted by easy early-depth episodes

    All training logic (Sense→Think→Act→Learn, Bellman updates, episode reset)
    stays inside TrainingAgentRunner.
    """

    def __init__(self, policy: ICubePolicy,
                 training_service: TrainingService,
                 scramble_len: int = 1,
                 save_every: int = 10_000):
        self._policy           = policy
        self._training_service = training_service
        self._save_every       = save_every

        self._runner = TrainingAgentRunner(
            policy=policy,
            scramble_len=scramble_len,
            max_steps=_max_steps_for_depth(scramble_len),
        )

        self._thread: threading.Thread | None = None
        self._stop_event  = threading.Event()
        self._stats_lock  = threading.Lock()

        # lifetime stats
        self._episodes:     int = 0
        self._solved_count: int = 0

        # curriculum state
        self._current_depth:    int   = scramble_len
        self._window_episodes:  int   = 0
        self._window_solved:    int   = 0
        self._stagnant_windows: int   = 0   # consecutive non-promoting windows

    # ------------------------------------------------------------------ stats (thread-safe reads)

    @property
    def episodes(self) -> int:
        with self._stats_lock:
            return self._episodes

    @property
    def solved_count(self) -> int:
        with self._stats_lock:
            return self._solved_count

    @property
    def current_depth(self) -> int:
        with self._stats_lock:
            return self._current_depth

    @property
    def window_solve_rate(self) -> float:
        """Live solve rate of the current evaluation window (resets every 500 episodes).
        More meaningful than lifetime rate which is diluted by easy early depths."""
        with self._stats_lock:
            if self._window_episodes == 0:
                return 0.0
            return self._window_solved / self._window_episodes

    def solve_rate(self) -> float:
        """Lifetime solve rate across all depths (kept for compatibility)."""
        with self._stats_lock:
            if self._episodes == 0:
                return 0.0
            return self._solved_count / self._episodes

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return (self._thread is not None
                and self._thread.is_alive()
                and not self._stop_event.is_set())

    # ------------------------------------------------------------------ loop (background thread only)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            result = self._runner.step()

            if not result.episode_done:
                continue

            # ---- per-episode accounting ----
            promote       = False
            boost_explore = False
            new_depth     = self._current_depth
            window_rate   = 0.0

            with self._stats_lock:
                self._episodes = result.episode_count
                if result.solved:
                    self._solved_count += 1

                self._window_episodes += 1
                if result.solved:
                    self._window_solved += 1

                # ---- evaluate window ----
                if self._window_episodes >= _WINDOW_SIZE:
                    window_rate = self._window_solved / self._window_episodes
                    self._window_episodes = 0
                    self._window_solved   = 0

                    if window_rate >= _PROMOTE_RATE and self._current_depth < _MAX_DEPTH:
                        # Agent is good enough — move to harder depth
                        self._current_depth  += 1
                        new_depth             = self._current_depth
                        self._stagnant_windows = 0
                        promote = True
                    else:
                        # Window complete but not promoting
                        self._stagnant_windows += 1
                        if self._stagnant_windows >= _MAX_STAGNANT_WINDOWS:
                            # Stuck too long — force exploration
                            self._stagnant_windows = 0
                            boost_explore = True

            # ---- apply curriculum changes outside the lock ----
            # (runner is only ever accessed from this background thread)
            if promote:
                self._runner.set_scramble_len(new_depth)
                self._runner.set_max_steps(_max_steps_for_depth(new_depth))
                self._policy.set_eps(_EPS_ON_PROMOTE)
                print(
                    f"[Curriculum] depth {new_depth - 1} → {new_depth}  "
                    f"(window rate: {window_rate:.1%})  "
                    f"eps reset to {_EPS_ON_PROMOTE}"
                )

            if boost_explore:
                self._policy.set_eps(_EPS_ON_STAGNANT)
                print(
                    f"[Curriculum] Plateau at depth {self._current_depth} "
                    f"after {_MAX_STAGNANT_WINDOWS} windows — "
                    f"boosting eps to {_EPS_ON_STAGNANT}"
                )

            # ---- periodic save ----
            if result.episode_count % self._save_every == 0:
                self._training_service.save_qtable()
                print(
                    f"[TrainingWorker] EP {result.episode_count}  "
                    f"eps={result.epsilon:.4f}  "
                    f"depth={self._current_depth}  "
                    f"win_rate={self.window_solve_rate:.1%}"
                )
