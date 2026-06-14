import sys
import os

# Ensure project root is on sys.path so all packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyray as pr
import numpy as np
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
    IDLE       = auto()  # cube stationary, waiting for user input
    SCRAMBLING = auto()  # scramble animation playing
    SOLVING    = auto()  # agent solving, one tick per completed animation
    SOLVED     = auto()  # last solve move queued — waiting for animation to finish


# ------------------------------------------------------------------ UI helpers

# Corner sign convention (X,Y,Z) -> corner index, matching cube_state.py:
# 0=UFR, 1=URB, 2=UBL, 3=ULF, 4=DFR, 5=DRB, 6=DBL, 7=DLF
_SIGNS_TO_CORNER = {
    (1, 1, 1): 0, (1, 1, -1): 1, (-1, 1, -1): 2, (-1, 1, 1): 3,
    (1, -1, 1): 4, (1, -1, -1): 5, (-1, -1, -1): 6, (-1, -1, 1): 7,
}

# render.cubes index = x*4 + y*2 + z (x,y,z in {0,1}), where 0 = negative side.
# Its "home" corner identity is whatever _SIGNS_TO_CORNER maps its initial signs to.
_RENDER_HOME_CORNER = {}
for _x in range(2):
    for _y in range(2):
        for _z in range(2):
            _idx = _x * 4 + _y * 2 + _z
            _signs = (1 if _x else -1, 1 if _y else -1, 1 if _z else -1)
            _RENDER_HOME_CORNER[_idx] = _SIGNS_TO_CORNER[_signs]


def _check_visual_vs_logical(render: Rubik, logical_state) -> None:
    """Compare render.cubes' geometric perm/orientation against the logical
    Cube2x2State and print any mismatches."""
    geom_perm = [None] * 8
    bad_orient = []
    for render_idx, cube in enumerate(render.cubes):
        center = cube[0].center
        signs = tuple(int(np.sign(round(c, 1))) or 1 for c in center)
        try:
            slot = _SIGNS_TO_CORNER[signs]
        except KeyError:
            print(f"[CHECK] piece {render_idx}: center {center} -> invalid signs {signs}")
            continue
        piece_identity = _RENDER_HOME_CORNER[render_idx]
        geom_perm[slot] = piece_identity

        orientation = cube[0].orientation
        tr = np.trace(orientation)
        angle = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))
        if logical_state.is_solved() and round(angle) != 0:
            bad_orient.append((render_idx, slot, piece_identity, round(angle, 1)))

    logical_perm = list(logical_state.perm)
    if geom_perm != logical_perm:
        print(f"[CHECK] PERM MISMATCH: geom={geom_perm} logical={logical_perm}")
    else:
        print(f"[CHECK] perm OK: {geom_perm}")

    if bad_orient:
        print(f"[CHECK] ORIENTATION MISMATCH (logical solved but visual not identity): {bad_orient}")
    else:
        print("[CHECK] orientation OK")


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
    ui_scramble_len   = 5   # depth used by the Randomize button (display cube)
    ui_max_steps      = 30  # step limit for the solving agent (display cube)

    # Core components — all shared state lives here, not in the UI
    policy = QLearningAgent()
    env_service = EnvironmentService(scramble_len=ui_scramble_len, max_steps=ui_max_steps)
    training_service = TrainingService(policy)
    solving_runner = SolvingAgentRunner(env_service, policy)
    training_worker = TrainingWorker(
        policy=policy,
        training_service=training_service,
        scramble_len=1,       # curriculum starts at depth 1 and auto-promotes
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
    _pending_done = None   # holds SolvingTickResult until final animation drains

    # Start with a scrambled cube
    _, scramble_seq = env_service.reset()
    for move in scramble_seq:
        rotation_queue.append(rubiks_moves[move])
    app_state = AppState.SCRAMBLING
    print("[CHECK] scramble:", scramble_seq)
    _solve_moves = []

    # ------------------------------------------------------------------ loop
    while not pr.window_should_close():
        animating = bool(rotation_queue) or render.is_rotating

        # ---------- input ----------

        # [Randomize] — reset cube and play new scramble
        if _btn_clicked(20, 150) and not animating:
            _pending_done = None
            render = Rubik()
            rotation_queue.clear()
            _, scramble_seq = env_service.reset()
            for move in scramble_seq:
                rotation_queue.append(rubiks_moves[move])
            app_state = AppState.SCRAMBLING
            status_text = ""
            print("[CHECK] scramble:", scramble_seq)
            _solve_moves = []

        # [Solve] — run the agent one tick at a time
        if _btn_clicked(190, 150) and not animating and app_state == AppState.IDLE:
            if training_service.qtable_exists() and not qtable_loaded:
                qtable_loaded = training_service.load_qtable()
            if qtable_loaded:
                env_service.prepare_for_solve()
                solving_runner.reset_visited()
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
                rotation_queue.append(rubiks_moves[result.action_str])
                _solve_moves.append(result.action_str)

                if result.done:
                    print("[CHECK] solve moves:", _solve_moves)
                    # Switch to SOLVED — defer the status message until the
                    # queued animation finishes so the cube visually matches
                    # the "Solved!" text when it appears.
                    app_state = AppState.SOLVED
                    _pending_done = result

        # ---------- scramble done ----------
        if app_state == AppState.SCRAMBLING and not animating:
            app_state = AppState.IDLE

        # ---------- animation ----------
        rotation_queue, _ = render.handle_rotation(rotation_queue)

        # ---------- reveal solve result after final animation drains ----------
        if app_state == AppState.SOLVED and not (bool(rotation_queue) or render.is_rotating):
            app_state = AppState.IDLE
            if _pending_done is not None:
                if _pending_done.is_solved:
                    status_text  = f"Solved in {_pending_done.step_count} steps!"
                    status_color = pr.GREEN
                    _check_visual_vs_logical(render, env_service.current_cube_state())
                else:
                    status_text  = f"Could not solve ({_pending_done.step_count} steps)."
                    status_color = pr.RED
                _pending_done = None

        # ---------- draw ----------
        if pr.is_mouse_button_down(pr.MouseButton.MOUSE_BUTTON_LEFT):
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
        pr.draw_rectangle(10, 10, 270, 194, pr.fade(pr.BLACK, 0.65))
        pr.draw_text(f"State:    {app_state.name}", 20, 18,  16, pr.RAYWHITE)
        pr.draw_text(f"Steps:    {env_service.step_count} / {ui_max_steps}", 20, 40, 16, pr.RAYWHITE)
        train_label = "ON" if training_worker.is_running() else "OFF"
        train_color = pr.GREEN if training_worker.is_running() else pr.RAYWHITE
        pr.draw_text(f"Training: {train_label}", 20, 62, 16, train_color)
        pr.draw_text(f"Episodes: {training_worker.episodes}", 20, 84, 16, pr.RAYWHITE)
        pr.draw_text(f"Depth:    {training_worker.current_depth} / 11", 20, 106, 16, pr.RAYWHITE)
        win_rate = training_worker.window_solve_rate
        win_color = pr.GREEN if win_rate >= 0.8 else pr.YELLOW if win_rate >= 0.4 else pr.RAYWHITE
        pr.draw_text(f"WinRate:  {win_rate:.1%}", 20, 128, 16, win_color)
        pr.draw_text(f"Epsilon:  {policy.eps:.4f}", 20, 150, 16, pr.RAYWHITE)
        pr.draw_text(f"Q-table:  {'loaded' if qtable_loaded else 'none'}", 20, 172, 16, pr.RAYWHITE)

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
