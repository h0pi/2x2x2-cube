# Concrete before/after changes

Each row is one change I can explain at the defense: what was wrong, what
changed, and in which file it's visible.

---

### 1. Orientation delta inconsistency (`cube_agent/domain/cube_state.py`)

**Symptom:** the agent would sometimes announce "Solved in N steps!" while the
3D cube still visibly had twisted corners.

**Before:**
```python
ORI_DELTA = {
    "U": (0, 0, 0, 0),
    "D": (0, 0, 0, 0),
    "F": (1, 2, 1, 2),
    "B": (1, 2, 1, 2),
    "R": (1, 2, 1, 2),
    "L": (1, 2, 1, 2),
}
```

**After:**
```python
ORI_DELTA = {
    "U": (0, 0, 0, 0),
    "D": (0, 0, 0, 0),
    "F": (1, 2, 1, 2),
    "B": (2, 1, 2, 1),
    "R": (2, 1, 2, 1),
    "L": (2, 1, 2, 1),
}
```

**Why:** F's permutation cycle (`MOVE_CYCLES["F"]`) has the opposite handedness
from B/R/L's, so giving all four faces the same orientation-delta pattern was
mathematically inconsistent — for certain real move sequences the logical
state reported `ori = (0,)*8` ("solved") while an independent geometric
(rotation-matrix) simulation showed 4 corners still twisted 120°. The fix was
verified by reproducing 5 real failing scramble+solve sequences and confirming
all 5 now correctly report `is_solved() == False`.
**Side effect:** this changes the state encoding, so the existing
`qtable_2x2.pkl` had to be evaluated for whether it needed retraining — see
note in `README.md`.

---

### 2. Q-table save was not atomic (`cube_agent/infrastructure/qtable_repository.py`)

**Before:**
```python
def save(qtable: dict, path: str = DEFAULT_PATH) -> None:
    """Persist Q-table dict to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(qtable, f)
    print(f"[Repository] Q-table saved to {path}  ({len(qtable)} entries)")
```

**After:**
```python
def save(qtable: dict, path: str = DEFAULT_PATH) -> None:
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(qtable, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    print(f"[Repository] Q-table saved to {path}  ({len(qtable)} entries)")
```

**Why:** an interruption mid-write (crash, force-quit, power loss) left a
half-written, unloadable `qtable_2x2.pkl`, destroying all prior training.
Writing to a temp file and using `os.replace()` (atomic at the filesystem
level) means the real path always holds either the complete old table or the
complete new one — never a partial write.

---

### 3. Curriculum logic lived in the UI layer (`cube_ui/workers/training_worker.py`)

**Before (excerpt):** the worker directly defined and used curriculum
constants and called setters on the policy/runner itself:
```python
_PROMOTE_RATE         = 0.80
_WINDOW_SIZE          = 500
_MAX_STAGNANT_WINDOWS = 5
_EPS_ON_PROMOTE       = 0.50
_EPS_ON_STAGNANT      = 0.30
...
if window_rate >= _PROMOTE_RATE and self._current_depth < _MAX_DEPTH:
    self._current_depth += 1
    self._runner.set_scramble_len(new_depth)
    self._runner.set_max_steps(_max_steps_for_depth(new_depth))
    self._policy.set_eps(_EPS_ON_PROMOTE)
```

**After:** all of the above moved into a new file,
`cube_agent/application/services/curriculum_service.py` (`CurriculumService`).
The worker now only does:
```python
with self._stats_lock:
    self._curriculum.record_episode(result.episode_count, result.solved)
    depth    = self._curriculum.current_depth
    win_rate = self._curriculum.window_solve_rate
```

**Why:** curriculum thresholds and promotion decisions are training/agent
policy, not UI concerns — per Clean Architecture, the UI layer must stay
"thin" and the training logic must be usable/testable with no GUI attached.
**Verified:** ran the curriculum logic standalone (no UI, no raylib) with 500
simulated solved episodes and confirmed it still promotes depth 1 → 2 at
100% window rate, and still boosts epsilon to 0.3 after 5 stagnant windows —
identical behavior to the original implementation.

---

### 4. No isolated proof that `update()` changes behavior (`demo_learning.py`, new file)

**Before:** `QLearningAgent.update()` (in `cube_agent/ml/qlearning_agent.py`)
existed and is called from `TrainingAgentRunner`, but nothing in the project
demonstrated its effect in isolation.

**After:** added `demo_learning.py` — fixes one cube state (solved cube + one
`U` move), prints its Q-values/greedy action before any training (all
zeros), runs 2,000 real training episodes on a **fresh, throwaway**
`QLearningAgent` (never touches the real `qtable_2x2.pkl`), then prints the
same state's Q-values/decision again.

**Observed output:**
```
--- BEFORE learning ---
U': Q = +0.0000   (tied with everything else; chosen action was arbitrary: D)

--- AFTER learning (2,000 episodes) ---
U': Q = +72.5433  <== greedy choice   (all other actions negative)
```

**Why:** this is the clearest, most direct way to show the reward signal from
the environment is actually changing the agent's behavior for a fixed input —
the core idea of the "Learn" step in Sense → Think → Act → Learn.

---

### 5. Documentation no longer matched the code (`README.md`)

**Before:** README stated "The UI never contains business logic" while
`training_worker.py` (UI layer) actually held curriculum thresholds and
called `set_eps`/`set_scramble_len`/`set_max_steps` directly — i.e. the claim
was false until fix #3 above.

**After:** once fix #3 made the claim true, README was updated to describe
`CurriculumService` in the application layer, describe `TrainingWorker` as
"daemon thread lifecycle + periodic saves only", and document fixes #1–#4 in
a "Notable implementation details" section.

**Why:** documentation should describe what the code actually does, not what
it used to do or what it ideally should do.
