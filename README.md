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
  infrastructure/     Q-table save/load (sparse pickle, atomic write via os.replace)
  application/
    services/         EnvironmentService (episode state, reward shaping)
                      TrainingService (Q-table persistence facade)
                      CurriculumService (depth promotion, plateau detection,
                                          max_steps scaling — works with or
                                          without a GUI attached)
    runners/          SolvingAgentRunner  — Sense → Think → Act  (+ loop prevention)
                      TrainingAgentRunner — Sense → Think → Act → Learn

cube_ui/              Thin host layer — raylib window, buttons, background thread
  workers/            TrainingWorker (daemon thread lifecycle + periodic saves only —
                      no curriculum thresholds; delegates decisions to CurriculumService)
  main.py             Window loop, AppState machine, button logic

run.py                Entry point — python run.py
demo_learning.py      Standalone script proving the Learn step changes behavior
                      (fresh in-memory agent, before/after Q-values for one fixed state)
```

Each runner tick is an explicit **Sense → Think → Act (→ Learn)** cycle.  
The UI never contains business logic; the agent never touches raylib. Curriculum
decisions (when to promote scramble depth, when to force exploration) live in
`CurriculumService` in the application layer, not in the UI worker — so the
training logic is testable and usable with no GUI at all.

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

**Camera:** click and drag with the left mouse button to orbit the view. Just hovering over the window no longer rotates the camera.

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

**Orientation delta fix** — `ORI_DELTA` previously gave F, B, R, and L the same `(1, 2, 1, 2)` pattern, but F's `MOVE_CYCLES` cycle has the opposite handedness from B/R/L's. With all four equal, some solve sequences ended with `ori = (0,)*8` (reported as solved) while 4 corners were still visually twisted 120°. Fixed by setting `B`, `R`, and `L` to `(2, 1, 2, 1)` while keeping `F` at `(1, 2, 1, 2)`. **This changes the state encoding — delete and retrain any existing `qtable_2x2.pkl` saved before this fix.**

**Curriculum logic moved out of the UI** — `TrainingWorker` (in `cube_ui/workers/`) used to own the curriculum constants (`_PROMOTE_RATE`, `_WINDOW_SIZE`, `_MAX_STAGNANT_WINDOWS`, `_EPS_ON_PROMOTE`, `_EPS_ON_STAGNANT`) and called `set_scramble_len()` / `set_max_steps()` / `set_eps()` directly — i.e. training decisions living in the UI layer, which broke the "UI has no business logic" rule. This logic now lives in `cube_agent/application/services/curriculum_service.py` (`CurriculumService`), which the worker simply delegates to after each finished episode. `CurriculumService` only depends on the runner/policy interfaces, so it works identically with or without a GUI attached. Verified to produce identical promotion/exploration behavior to the original implementation.

**Atomic Q-table saves** — `qtable_repository.save()` previously wrote directly to `qtable_2x2.pkl`; an interrupted write (crash, force-quit) would leave a corrupt, unloadable file and lose all prior training. It now writes to `qtable_2x2.pkl.tmp` first, then calls `os.replace()` onto the final path — atomic at the filesystem level, so the file is always either the complete old table or the complete new one, never a partial write.

**Learning demo** — `demo_learning.py` is a standalone script that proves the Bellman update actually changes agent behavior: it fixes one cube state, prints its Q-values and greedy action before any training (all zeros — an untrained, effectively arbitrary choice), runs 2,000 real training episodes, then prints the same state's Q-values and decision again (now clearly preferring the correct solving move). It uses a fresh, in-memory `QLearningAgent` and never touches the project's real `qtable_2x2.pkl`.

---

## Project layout

```
2x2x2-cube/
├── run.py
├── demo_learning.py                 ← standalone proof that Learn changes behavior
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
│   │   └── qtable_repository.py    Sparse pickle save/load, atomic write (os.replace)
│   └── application/
│       ├── services/
│       │   ├── environment_service.py
│       │   ├── training_service.py
│       │   └── curriculum_service.py     Depth promotion, plateau detection,
│       │                                 max_steps scaling (GUI-independent)
│       └── runners/
│           ├── solving_agent_runner.py   greedy + visited-set loop prevention
│           └── training_agent_runner.py  epsilon-greedy + Bellman update
└── cube_ui/
    ├── configs.py                  Window size, FPS, rubiks_moves rotation table
    ├── rubik.py                    3D cube renderer (raylib)
    ├── main.py                     Window loop, AppState, buttons
    └── workers/
        └── training_worker.py      Daemon thread lifecycle + periodic saves only
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
