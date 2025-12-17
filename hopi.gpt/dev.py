import pyray as pr
import configs
from rubik import Rubik
import pickle
from env2x2 import Rubik2x2Env
from qlearning_agent import QLearningAgent
from moves2x2 import ACTIONS, IDX_TO_ACTION
from collections import defaultdict

pr.init_window(configs.window_w, configs.window_h, "Rubik 2x2 RL")
pr.set_target_fps(configs.fps)

env = Rubik2x2Env(scramble_len=2, max_steps=30)
agent = QLearningAgent(ACTIONS)
with open("qtable_2x2.pkl", "rb") as f:
    loaded_Q = pickle.load(f)

agent.Q = defaultdict(float, loaded_Q)

agent.eps = 0.0

#TRAIN_STEPS_PER_FRAME = 100

state, scramble_seq = env.reset()
render = Rubik() 
rotation_queue=[]
for move in scramble_seq:
    rotation_queue.append(configs.rubiks_moves[move])
pending_action_idx = None

episode = 0
ep_reward = 0
step_count = 0
last_solved = False
is_scrambling = True


while not pr.window_should_close():

    # ako render ne rotira i nemamo pending akciju -> agent bira potez
    if (not render.is_rotating) and (pending_action_idx is None) and not rotation_queue:
        a = agent.choose_action(state)
        move_str = IDX_TO_ACTION[a]
        rotation_queue.append(configs.rubiks_moves[move_str])  # animacija
        pending_action_idx = a

    # odradi animaciju
    rotation_queue, _ = render.handle_rotation(rotation_queue)

    # kad animacija poteza završi -> env step + Q update
    if pending_action_idx is not None and (not render.is_rotating) and not rotation_queue:
        is_scrambling = False
        print("[DEV] Scramble finished, starting agent")

        move_str = IDX_TO_ACTION[pending_action_idx]

        # JEDAN KORAK – BEZ UČENJA
        state, reward, done = env.step(move_str)

        #state = next_state
        step_count += 1
        last_solved = env.is_solved()

        pending_action_idx = None

        if last_solved or done:
            print(f"[DEMO] Solved in {step_count} steps")

            # RESET ENV + RENDER S ISTIM SCRAMBLE-OM
            state, scramble_seq = env.reset()
            render = Rubik()
            rotation_queue.clear()

            for move in scramble_seq:
                rotation_queue.append(configs.rubiks_moves[move])

            step_count = 0
            #last_solved = False
            #pending_action_idx = None
            episode += 1

        

    pr.update_camera(configs.camera, pr.CameraMode.CAMERA_THIRD_PERSON)

    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)
    pr.begin_mode_3d(configs.camera)

    for cube in render.cubes:
        for cube_part in cube:
            position = pr.Vector3(cube[0].center[0], cube[0].center[1], cube[0].center[2])
            pr.draw_model(cube_part.model, position, 2, cube_part.face_color)

    pr.draw_grid(20, 1.0)
    pr.end_mode_3d()
    # --- HUD / INFO ---
    pr.draw_rectangle(10, 10, 260, 140, pr.fade(pr.BLACK, 0.6))

    pr.draw_text(f"Episode: {episode}", 20, 20, 18, pr.RAYWHITE)
    pr.draw_text(f"Steps: {step_count} / {env.max_steps}", 20, 45, 18, pr.RAYWHITE)
    #pr.draw_text(f"Reward: {ep_reward:.2f}", 20, 70, 18, pr.RAYWHITE)
    pr.draw_text(f"Epsilon: {agent.eps:.3f}", 20, 95, 18, pr.RAYWHITE)
    pr.draw_text(
        f"Solved (env): {'YES' if last_solved else 'NO'}",
        20,
        120,
        18,
        pr.GREEN if last_solved else pr.RED
    )

    pr.end_drawing()

pr.close_window()
