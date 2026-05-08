# 2×2×2 Rubik's Cube — Tabular Q-Learning Agent

> A reinforcement learning agent that learns to solve a 2×2×2 Rubik's cube from scratch,  
> with real-time 3D visualization and a Clean Architecture design.

---

## What it does

The agent observes the current cube state, picks a move, and learns from the reward signal — over and over, thousands of times per second in a background thread.  
Once trained, you click **Solve** and watch it unscramble the cube step-by-step in 3D.

---

## Features

- **Tabular Q-learning** with epsilon-greedy exploration and Bellman updates
- **Real-time 3D visualization** powered by [raylib](https://www.raylib.com/) (via `raylib` Python bindings)
- **Background training thread** — train and watch the cube rotate at the same time, no UI freeze
- **Three UI buttons** — `Randomize`, `Solve`, `Train ON/OFF`
- **Live HUD** — episode count, epsilon, solve rate, Q-table status
- **Q-table persistence** — saved every 10 000 episodes, reloaded on next launch
- **Clean Architecture** — domain, application, infrastructure, and UI are strictly separated

---

## Architecture

```
agents_core/          Generic agent abstractions (SoftwareAgent, interfaces)
                      No domain knowledge — reusable across any RL project

cube_agent/           The "brain" — all logic, no pyray, no threads
  domain/             Cube state machine, move tables, reward rules
  ml/                 Q-learning policy (thread-safe RLock), ICubePolicy interface
  infrastructure/     Q-table save/load (pickle)
  application/
    services/         EnvironmentService (episode state, reward shaping)
                      TrainingService (Q-table persistence facade)
    runners/          SolvingAgentRunner  — Sense → Think → Act
                      TrainingAgentRunner — Sense → Think → Act → Learn

cube_ui/              Thin host layer — pyray window, buttons, background thread
  workers/            TrainingWorker (daemon thread, periodic saves)
  main.py             Window loop, AppState machine, button logic

run.py                Entry point — python run.py
```

Each runner tick is an explicit **Sense → Think → Act (→ Learn)** cycle.  
The UI never contains business logic; the agent never touches pyray.

---

## State space

| Dimension | Size |
|-----------|------|
| Corner permutations (8!) | 40 320 |
| Corner orientations (3⁷, last constrained) | 2 187 |
| **Total reachable states** | **~3.7 million** |

State is encoded as a single integer: `perm_factoradic × 6561 + ori_base3`.  
The Q-table is a `defaultdict(float)` keyed by `(state, action_index)`.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate α | 0.2 |
| Discount γ | 0.95 |
| Initial ε | 1.0 |
| Minimum ε | 0.05 |
| ε decay (per episode) | × 0.995 |
| Scramble depth | 5 moves |
| Max steps per episode | 30 |
| Reward per step | −0.1 |
| Reward for progress | ±Δheuristic × 2.0 |
| Reward on solve | +100 |

---

## Installation

```bash
git clone https://github.com/<your-username>/2x2x2-cube.git
cd 2x2x2-cube
pip install raylib numpy
```

Python 3.11+ recommended.

---

## Running

```bash
python run.py
```

### Controls

| Button | What it does |
|--------|--------------|
| **Randomize** | Scramble the cube (5 random moves) and reset the agent's episode |
| **Train ON/OFF** | Start or stop the background training thread |
| **Solve** | Let the trained agent solve the cube one move at a time |

**Typical workflow:**

1. Launch the app — the cube starts scrambled automatically
2. Click **Train ON** and leave it running (the episode counter climbs in the HUD)
3. After enough episodes (50k–200k depending on hardware), click **Train OFF**
4. Click **Solve** — the agent plays back its learned policy on the current cube
5. Click **Randomize** to try a new scramble and **Solve** again

> **Training time note:** the full state space has ~3.7 million reachable states.  
> On a laptop CPU, ~400k episodes yields roughly a 37 % solve rate — expect 800k–1M+  
> episodes for reliable solving. This takes several hours of background training across  
> multiple sessions. The Q-table is saved automatically every 10 000 episodes and  
> reloaded on the next run, so you can close the app and continue where you left off.

---

## Project layout

```
2x2x2-cube/
├── run.py
├── qtable_2x2.pkl              ← auto-generated after training
├── agents_core/
│   └── base_agent.py           SoftwareAgent[TPercept,TAction,TResult,TExperience]
├── cube_agent/
│   ├── domain/
│   │   ├── cube_state.py       Cube2x2State, MOVE_CYCLES, ORI_DELTA
│   │   ├── actions.py          ACTIONS list, IDX_TO_ACTION, ACTION_TO_IDX
│   │   └── results.py          SolvingTickResult, TrainingTickResult
│   ├── ml/
│   │   ├── i_cube_policy.py    ICubePolicy interface
│   │   └── qlearning_agent.py  Thread-safe tabular Q-learning
│   ├── infrastructure/
│   │   └── qtable_repository.py  Pickle save/load
│   └── application/
│       ├── services/
│       │   ├── environment_service.py
│       │   └── training_service.py
│       └── runners/
│           ├── solving_agent_runner.py
│           └── training_agent_runner.py
└── cube_ui/
    ├── configs.py              Window size, FPS, rubiks_moves rotation table
    ├── rubik.py                3D cube renderer (raylib)
    ├── main.py                 Window loop, AppState, buttons
    └── workers/
        └── training_worker.py  Background daemon thread
```

---

## Tech stack

- **Python 3.11**
- **raylib** — 3D rendering (`pip install raylib`)
- **NumPy** — rotation matrices for animation
- **threading** — background training with `RLock`-protected Q-table
- **pickle** — Q-table persistence

---

## License

MIT
