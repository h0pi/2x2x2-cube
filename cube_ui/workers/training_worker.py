import threading

from cube_agent.application.runners.training_agent_runner import TrainingAgentRunner
from cube_agent.application.services.curriculum_service import (
    CurriculumService, max_steps_for_depth,
)
from cube_agent.application.services.training_service import TrainingService
from cube_agent.ml.i_cube_policy import ICubePolicy


class TrainingWorker:
    """
    Background thread that drives the TrainingAgentRunner in a tight loop.

    Pure plumbing — no curriculum/training decisions live here. It only:
      - manages the threading lifecycle (start / stop)
      - calls runner.step() in a loop and forwards finished episodes to
        CurriculumService (which owns all promotion/exploration decisions)
      - triggers periodic Q-table saves
      - exposes CurriculumService's stats for the UI HUD

    All training logic (Sense→Think→Act→Learn, Bellman updates, episode
    reset) stays inside TrainingAgentRunner; all curriculum logic (depth
    promotion, plateau detection, max_steps scaling) stays inside
    CurriculumService — both work the same with or without a GUI attached.
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
            max_steps=max_steps_for_depth(scramble_len),
        )
        self._curriculum = CurriculumService(
            runner=self._runner, policy=policy, start_depth=scramble_len,
        )

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------ stats (thread-safe reads)

    @property
    def episodes(self) -> int:
        with self._stats_lock:
            return self._curriculum.episodes

    @property
    def solved_count(self) -> int:
        with self._stats_lock:
            return self._curriculum.solved_count

    @property
    def current_depth(self) -> int:
        with self._stats_lock:
            return self._curriculum.current_depth

    @property
    def window_solve_rate(self) -> float:
        with self._stats_lock:
            return self._curriculum.window_solve_rate

    def solve_rate(self) -> float:
        with self._stats_lock:
            return self._curriculum.solve_rate()

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

            with self._stats_lock:
                self._curriculum.record_episode(result.episode_count, result.solved)
                depth     = self._curriculum.current_depth
                win_rate  = self._curriculum.window_solve_rate

            if result.episode_count % self._save_every == 0:
                self._training_service.save_qtable()
                print(
                    f"[TrainingWorker] EP {result.episode_count}  "
                    f"eps={result.epsilon:.4f}  "
                    f"depth={depth}  "
                    f"win_rate={win_rate:.1%}"
                )
