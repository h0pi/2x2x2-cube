import threading

from cube_agent.application.runners.training_agent_runner import TrainingAgentRunner
from cube_agent.application.services.training_service import TrainingService
from cube_agent.ml.i_cube_policy import ICubePolicy

# Curriculum settings
_MAX_DEPTH       = 11    # God's number for 2×2×2
_PROMOTE_RATE    = 0.80  # solve rate needed to advance to next depth
_WINDOW_SIZE     = 500   # evaluate over this many episodes before deciding


class TrainingWorker:
    """
    Background thread that drives the TrainingAgentRunner in a tight loop.

    Manages:
      - threading lifecycle (start / stop)
      - curriculum learning: starts at depth 1, promotes every _WINDOW_SIZE
        episodes once the solve rate exceeds _PROMOTE_RATE, up to depth 11
      - periodic Q-table saves
      - stats counters (episodes, solved count, current depth) for the HUD

    All training logic (Sense->Think->Act->Learn, epsilon decay,
    reward shaping, episode reset) stays inside TrainingAgentRunner.
    """

    def __init__(self, policy: ICubePolicy,
                 training_service: TrainingService,
                 scramble_len: int = 1,
                 max_steps: int = 30,
                 save_every: int = 10_000):
        self._policy = policy
        self._training_service = training_service
        self._save_every = save_every
        self._runner = TrainingAgentRunner(
            policy=policy,
            scramble_len=scramble_len,
            max_steps=max_steps,
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()

        # lifetime stats
        self._episodes: int = 0
        self._solved_count: int = 0

        # curriculum state
        self._current_depth: int = scramble_len
        self._window_episodes: int = 0
        self._window_solved: int = 0

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

    def solve_rate(self) -> float:
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

            if result.episode_done:
                promote = False
                new_depth = self._current_depth

                with self._stats_lock:
                    self._episodes = result.episode_count
                    if result.solved:
                        self._solved_count += 1

                    # curriculum window tracking
                    self._window_episodes += 1
                    if result.solved:
                        self._window_solved += 1

                    if self._window_episodes >= _WINDOW_SIZE:
                        window_rate = self._window_solved / self._window_episodes
                        if window_rate >= _PROMOTE_RATE and self._current_depth < _MAX_DEPTH:
                            self._current_depth += 1
                            new_depth = self._current_depth
                            promote = True
                        # reset window regardless (re-evaluate next window)
                        self._window_episodes = 0
                        self._window_solved = 0

                # promote outside the lock (runner access is background-thread-only)
                if promote:
                    self._runner.set_scramble_len(new_depth)
                    print(f"[Curriculum] → depth {new_depth}  "
                          f"(window solve rate: {window_rate:.1%})")

                if result.episode_count % self._save_every == 0:
                    self._training_service.save_qtable()
                    print(
                        f"[TrainingWorker] EP {result.episode_count}  "
                        f"eps={result.epsilon:.4f}  "
                        f"depth={self._current_depth}  "
                        f"solve_rate={self.solve_rate():.2%}"
                    )
