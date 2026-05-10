# 2×2×2 Rubik's Cube — Tabular Q-Learning Agent

> A reinforcement learning agent that learns to solve a 2×2×2 Rubik's cube from scratch,  
> with real-time 3D visualization and a Clean Architecture design.

---

## What it does

The agent observes the current cube state, picks a move, and learns from the reward signal — thousands of times per second in a background thread.  
It starts by learning to undo a single scramble move, then progressively tackles harder scrambles through curriculum learning.  
Once trained, you click **Solve** and watch it unscramble the cube step-by-step in 3D.

---

## Features

- **Tabular Q-learning** with epsilon-greedy exploration and Bellman updates
- **numpy float32 Q-table** — ~50× less RAM than a Python dict; demand-paged so physical memory stays low
- **Curriculum learning** — training starts at scramble depth 1 and auto-promotes up to depth 11 (God's number) as solve rate improves
- **Loop prevention** — the solving agent tracks visited states and never revisits them in a single solve attempt
- **Real-time 3D visualization** powered by [raylib](https://www.raylib.com/) (via `raylib` Python bindings)
- **Background training thread** — train and watch the cube rotate at the same time, no UI freeze
- **Three UI buttons** — `Randomize`, `Solve`, `Train ON/OFF`
- **Live HUD** — episode count, curriculum depth, epsilon, Q-table status
- **Q-table persistence** — saved every 10 000 episodes, reloaded on next launch; corrupt saves handled gracefully
- **Clean Architecture** — domain, application, infrastructure, and UI are strictly separated

---

## Architecture

```
agents_core/          Generic agent abstractions (SoftwareAgent, interfaces)
                      No domain knowledge — reusable across any RL project

cube_agent/           The "brain" — all logic, no pyray, no threads
  domain/             Cube state machine, move tables, reward rules
  ml/                 numpy Q-table (thread-safe RLock), ICubePolicy interface
  infrastructure/     Q-table save/load (sparse pickle)
  application/
    services/         EnvironmentService (episode state, reward shaping)
                      TrainingService (Q-table persistence facade)
    runners/          SolvingAgentRunner  — Sense → Think → Act  (+ loop prevention)
                      TrainingAgentRunner — Sense → Think → Act → Learn

cube_ui/              Thin host layer — raylib window, buttons, background thread
  workers/            TrainingWorker (daemon thread, curriculum promotion, periodic saves)
  main.py             Window loop, AppState machine, button logic

run.py                Entry point — python run.py
```

Each runner tick is an explicit **Sense → Think → Act (→ Learn)** cycle.  
The UI never contains business logic; the agent never touches raylib.

---

## State encoding

| Dimension | Size |
|-----------|------|
| Corner permutations (8!) | 40 320 |
| Corner orientations (3⁷, 8th determined by sum ≡ 0 mod 3) | 2 187 |
| **Total encoded state space** | **88 179 840** |

State is encoded as `perm_factoradic × 2187 + ori_base3` (first 7 orientations only).  
The Q-table is a **numpy float32 array** of shape `(88 179 840, 12)` — ~4.2 GB virtual address space, demand-paged. Physical RAM usage is proportional to the states actually visited during training, typically well under 1 GB in early sessions.

Saves are **sparse**: only non-zero entries are pickled (`rows`, `cols`, `values` arrays), keeping file sizes small regardless of array size.

---

## Curriculum learning

Training automatically increases scramble difficulty as the agent improves:

| Setting | Value |
|---------|-------|
| Starting depth | 1 move |
| Maximum depth | 11 moves (God's number for 2×2×2) |
| Promotion threshold | 80 % solve rate |
| Evaluation window | 500 episodes |

The HUD shows `Depth: X / 11` in real time. Each depth level is only unlocked after the agent can reliably solve the simpler version, so training time is spent efficiently.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate α | 0.2 |
| Discount γ | 0.95 |
| Initial ε | 1.0 |
| Minimum ε | 0.05 |
| ε decay (per episode) | × 0.995 |
| Display scramble depth | 5 moves |
| Max steps per solve episode | 30 |
| Reward per step | −0.1 |
| Reward for progress | ±Δheuristic × 2.0 |
| Reward on solve | +100 |

---

## Installation

```bash
git clone https://github.com/h0pi/2x2x2-cube.git
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
| **Randomize** | Scramble the display cube (5 moves, no cancelling sequences) and reset the episode |
| **Train ON/OFF** | Start or stop the background curriculum training thread |
| **Solve** | Let the trained agent solve the current cube one move at a time |

**Typical workflow:**

1. Launch the app — the cube starts scrambled automatically
2. Click **Train ON** and leave it running (watch the episode counter and depth climb in the HUD)
3. Let it train across multiple sessions — the Q-table reloads from disk automatically on each launch
4. Click **Train OFF** then **Solve** to watch the agent work

> **Training time note:** on a laptop CPU, reaching depth 5 with ~60 % solve rate takes roughly  
> 800k–900k episodes (~several hours). Higher depths require more. The curriculum approach  
> means early training is fast (depth 1–3 trains in minutes) and difficulty ramps gradually.

---

## Notable implementation details

**Orientation tracking fix** — the original model was missing a `_cycle4(ori, …)` call inside `apply()`, meaning corner orientations were not carried along when pieces moved. This made every 4th application of the same face move return wrong orientations (F⁴ ≠ identity in the broken model), corrupting the entire state space. Fixed by adding the missing cycle before the delta update.

**Scramble quality** — `scramble()` filters out moves on the same face as the previous move, preventing sequences like `F F'` or `R R` that waste scramble depth and produce easier-than-intended positions.

**Loop prevention** — `SolvingAgentRunner` maintains a visited-state set per solve attempt. At each step it sorts all 12 actions by Q-value and picks the best one whose resulting state has not yet been visited. This eliminates oscillation between two states when Q-values are tied or undertrained.

**Memory efficiency** — replacing `defaultdict` with a numpy array reduced peak RAM from ~18 GB (Python dict overhead: ~200 bytes/entry) to under 1 GB physical during typical training (numpy: 4 bytes/entry, demand-paged).

---

## Project layout

```
2x2x2-cube/
├── run.py
├── qtable_2x2.pkl                  ← auto-generated after training
├── agents_core/
│   └── base_agent.py               SoftwareAgent[TPercept,TAction,TResult,TExperience]
├── cube_agent/
│   ├── domain/
│   │   ├── cube_state.py           Cube2x2State, MOVE_CYCLES, ORI_DELTA, scramble
│   │   ├── actions.py              ACTIONS list, IDX_TO_ACTION, ACTION_TO_IDX
│   │   └── results.py              SolvingTickResult, TrainingTickResult
│   ├── ml/
│   │   ├── i_cube_policy.py        ICubePolicy interface
│   │   └── qlearning_agent.py      numpy-backed thread-safe Q-learning
│   ├── infrastructure/
│   │   └── qtable_repository.py    Sparse pickle save/load with corruption handling
│   └── application/
│       ├── services/
│       │   ├── environment_service.py
│       │   └── training_service.py
│       └── runners/
│           ├── solving_agent_runner.py   greedy + visited-set loop prevention
│           └── training_agent_runner.py  epsilon-greedy + Bellman update
└── cube_ui/
    ├── configs.py                  Window size, FPS, rubiks_moves rotation table
    ├── rubik.py                    3D cube renderer (raylib)
    ├── main.py                     Window loop, AppState, buttons
    └── workers/
        └── training_worker.py      Daemon thread + curriculum promotion logic
```

---

## Tech stack

- **Python 3.11**
- **raylib** — 3D rendering (`pip install raylib`)
- **NumPy** — Q-table storage and animation rotation matrices
- **threading** — background training with `RLock`-protected Q-table
- **pickle** — sparse Q-table persistence

---

## License

MIT
