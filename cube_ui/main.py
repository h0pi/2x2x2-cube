import sys
import os

# Ensure project root is on sys.path so all packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyray as pr
from enum import Enum, auto

from cube_agent.application.services.environment_service import EnvironmentService
from cube_agent.application.services.training_service import TrainingService
from cube_agent.application.runners.solving_agent_runner import SolvingAgentRunner
from cube_agent.ml.qlearning_agent import QLearningAgent
from cube_ui.rubik import Rubik
from cube_ui.configs import window_w, window_h, fps, rubiks_moves
from cube_ui.workers.training_worker import TrainingWorker


# ------------------------------------------------------------------ app state

class AppState(Enum):
    IDLE = auto()       # cube stationary, waiting for user input
    SCRAMBLING = auto() # scramble animation playing
    SOLVING = auto()    # agent solving, one tick per completed animation


# ------------------------------------------------------------------ UI helpers

_BTN_H = 40
_BTN_Y = window_h - 60


def _draw_button(label: str, x: int, w: int,
                 enabled: bool = True, active: bool = False) -> None:
    if active:
        bg = pr.Color(30, 180, 30, 255)
        fg = pr.BLACK
    elif not enabled:
        bg = pr.Color(80, 80, 80, 255)
        fg = pr.Color(150, 150, 150, 255)
    else:
        bg = pr.Color(50, 50, 200, 255)
        fg = pr.WHITE

    pr.draw_rectangle(x, _BTN_Y, w, _BTN_H, bg)
    pr.draw_rectangle_lines(x, _BTN_Y, w, _BTN_H, pr.WHITE)
    text_w = pr.measure_text(label, 16)
    pr.draw_text(label, x + (w - text_w) // 2, _BTN_Y + (_BTN_H - 16) // 2, 16, fg)


def _btn_clicked(x: int, w: int) -> bool:
    if not pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_LEFT):
        return False
    mp = pr.get_mouse_position()
    return pr.check_collision_point_rec(mp, pr.Rectangle(x, _BTN_Y, w, _BTN_H))


# ------------------------------------------------------------------ main

def main() -> None:
    scramble_len = 5
    max_steps = 30

    # Core components — all shared state lives here, not in the UI
    policy = QLearningAgent()
    env_service = EnvironmentService(scramble_len=scramble_len, max_steps=max_steps)
    training_service = TrainingService(policy)
    solving_runner = SolvingAgentRunner(env_service, policy)
    training_worker = TrainingWorker(
        policy=policy,
        training_service=training_service,
        scramble_len=scramble_len,
        max_steps=max_steps,
        save_every=10_000,
    )

    qtable_loaded = training_service.load_qtable()

    pr.init_window(window_w, window_h, "Rubik 2x2 RL Agent")
    pr.set_target_fps(fps)

    # Camera must be created after init_window()
    camera = pr.Camera3D(
        pr.Vector3(18.0, 16.0, 18.0),
        pr.Vector3(0.0, 0.0, 0.0),
        pr.Vector3(0.0, 1.0, 0.0),
        45.0,
        pr.CameraProjection.CAMERA_PERSPECTIVE,
    )

    render = Rubik()
    rotation_queue: list = []
    app_state = AppState.IDLE
    status_text = ""
    status_color = pr.RAYWHITE

    # Start with a scrambled cube
    _, scramble_seq = env_service.reset()
    for move in scramble_seq:
        rotation_queue.append(rubiks_moves[move])
    app_state = AppState.SCRAMBLING

    # ------------------------------------------------------------------ loop
    while not pr.window_should_close():
        animating = bool(rotation_queue) or render.is_rotating

        # ---------- input ----------

        # [Randomize] — reset cube and play new scramble
        if _btn_clicked(20, 150) and not animating:
            render = Rubik()
            rotation_queue.clear()
            _, scramble_seq = env_service.reset()
            for move in scramble_seq:
                rotation_queue.append(rubiks_moves[move])
            app_state = AppState.SCRAMBLING
            status_text = ""

        # [Solve] — run the agent one tick at a time
        if _btn_clicked(190, 150) and not animating and app_state == AppState.IDLE:
            if training_service.qtable_exists() and not qtable_loaded:
                qtable_loaded = training_service.load_qtable()
            if qtable_loaded:
                env_service.prepare_for_solve()
                app_state = AppState.SOLVING
                status_text = ""
            else:
                status_text = "No Q-table found — train first!"
                status_color = pr.RED

        # [Train ON/OFF] — toggle background training thread
        if _btn_clicked(360, 160):
            if training_worker.is_running():
                training_worker.stop()
                training_service.save_qtable()
                qtable_loaded = True
                status_text = "Training stopped — Q-table saved."
                status_color = pr.YELLOW
            else:
                training_worker.start()
                status_text = "Training started."
                status_color = pr.GREEN

        # ---------- solving tick (one move per completed animation) ----------
        if app_state == AppState.SOLVING and not animating:
            result = solving_runner.step()

            if result is None:
                app_state = AppState.IDLE
                status_text = "Already solved!"
                status_color = pr.GREEN
            else:
                # Queue the animation the runner already executed in env
                rotation_queue.append(rubiks_moves[result.action_str])

                if result.done:
                    app_state = AppState.IDLE
                    if result.is_solved:
                        status_text = f"Solved in {result.step_count} steps!"
                        status_color = pr.GREEN
                    else:
                        status_text = f"Could not solve ({result.step_count} steps)."
                        status_color = pr.RED

        # ---------- scramble done ----------
        if app_state == AppState.SCRAMBLING and not animating:
            app_state = AppState.IDLE

        # ---------- animation ----------
        rotation_queue, _ = render.handle_rotation(rotation_queue)

        # ---------- draw ----------
        pr.update_camera(camera, pr.CameraMode.CAMERA_THIRD_PERSON)
        pr.begin_drawing()
        pr.clear_background(pr.RAYWHITE)

        pr.begin_mode_3d(camera)
        for cube in render.cubes:
            for part in cube:
                pos = pr.Vector3(cube[0].center[0], cube[0].center[1], cube[0].center[2])
                pr.draw_model(part.model, pos, 2, part.face_color)
        pr.draw_grid(20, 1.0)
        pr.end_mode_3d()

        # HUD — top left
        pr.draw_rectangle(10, 10, 270, 150, pr.fade(pr.BLACK, 0.65))
        pr.draw_text(f"State:    {app_state.name}", 20, 18,  16, pr.RAYWHITE)
        pr.draw_text(f"Steps:    {env_service.step_count} / {max_steps}", 20, 40, 16, pr.RAYWHITE)
        train_label = "ON" if training_worker.is_running() else "OFF"
        train_color = pr.GREEN if training_worker.is_running() else pr.RAYWHITE
        pr.draw_text(f"Training: {train_label}", 20, 62, 16, train_color)
        pr.draw_text(f"Episodes: {training_worker.episodes}", 20, 84, 16, pr.RAYWHITE)
        pr.draw_text(f"Epsilon:  {policy.eps:.4f}", 20, 106, 16, pr.RAYWHITE)
        pr.draw_text(f"Q-table:  {'loaded' if qtable_loaded else 'none'}", 20, 128, 16, pr.RAYWHITE)

        # Status line above buttons
        if status_text:
            pr.draw_text(status_text, 20, _BTN_Y - 28, 17, status_color)

        # Buttons
        _draw_button("Randomize", 20,  150, enabled=not animating)
        can_solve = (app_state == AppState.IDLE and not animating
                     and (qtable_loaded or training_service.qtable_exists()))
        _draw_button("Solve", 190, 150, enabled=can_solve)
        _draw_button(
            f"Train: {'ON ' if training_worker.is_running() else 'OFF'}",
            360, 160,
            active=training_worker.is_running(),
        )

        pr.end_drawing()

    # ---------- shutdown ----------
    training_worker.stop()
    if training_worker.episodes > 0:
        training_service.save_qtable()
        print(f"[Main] Shutdown — saved Q-table after {training_worker.episodes} episodes.")
    pr.close_window()


if __name__ == '__main__':
    main()
