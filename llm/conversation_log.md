# Condensed conversation log

Two sessions, condensed to the decision-relevant exchanges. Full prompts are
in `prompts.md`; this file shows the reasoning flow.

---

## Session A — "Solved but visually wrong" bug

**Me:** The agent says "Solved in N steps!" but the rendered cube still looks
twisted. Happens sometimes, not always. Find the issue, don't change code yet.

**LLM:** Added non-destructive diagnostics first: `[CHECK]` print statements
in `cube_ui/main.py` that, right when the UI shows "Solved!", compare the
3D-rendered cube's actual geometry (corner positions + rotation matrices)
against the logical `Cube2x2State` the agent thinks it has. No behavior
changed yet — just visibility.

**Me:** (pasted real console output) — perm matched, but orientation showed
exactly 4 corners at ~120° rotation even though the logical state said
`is_solved() == True`.

**LLM:** Took the exact failing move sequence from that log and reproduced it
in an offline geometric simulation (independent rotation-matrix math, not
reusing the app's own logic) to confirm the mismatch was real and
reproducible outside the live app. Then brute-forced every combination of the
4 movable faces' `ORI_DELTA` patterns against that sequence, found that only
combinations where F's pattern differs from L's pattern matched the true
geometry.

**Me:** (after the first, partial fix) Pasted four more real failure logs
from different scrambles (using B, R, U, D, F moves — not just D/F/L).

**LLM:** Re-ran the same brute-force search using all 5 real failing
sequences together, found the one consistent table — F keeps `(1,2,1,2)`,
B/R/L all become `(2,1,2,1)` — verified it against all 5 sequences (0
mismatches) plus a basic group sanity check (applying any face 4 times must
return to solved).

**Outcome:** `cube_agent/domain/cube_state.py`'s `ORI_DELTA` updated; noted
the state encoding changes, so README documents that the existing Q-table
should be retrained if affected.

---

## Session B — Acting on professor feedback

**Me:** Pasted the professor's feedback (Bosnian) listing 5 required fixes.
Asked for a plain breakdown before touching anything.

**LLM:** Summarized the 5 items in plain language: (1) `llm/` folder with
proof of LLM usage, (2) an isolated learning demo, (3) move curriculum logic
out of the UI layer, (4) atomic Q-table save, (5) sync docs with code. Noted
none of them require retraining.

**Me:** "yes" — go ahead, starting with the smallest/safest.

**LLM:** Fixed atomic save first (`qtable_repository.py`): write to
`.tmp`, `os.replace()` onto the real path. Verified with a throwaway test
script in a temp directory (not the real Q-table) that no `.tmp` file is
left behind and the saved content loads back correctly.

**Me:** Confirmed, move to curriculum refactor.

**LLM:** Read `training_worker.py` and `training_agent_runner.py` to
understand existing behavior, then extracted the constants and the
promote/plateau decision logic into a new
`cube_agent/application/services/curriculum_service.py` (`CurriculumService`),
leaving `training_worker.py` as pure thread lifecycle + delegation. Verified
with a standalone script (no UI/raylib) that simulating 500 solved episodes
still promotes depth 1→2 at 100% rate, and 5 stagnant windows still boosts
epsilon to 0.3 — same as before the refactor.

**Me:** "what about second issue? how to do that" then "do i need to retrain
my agent bcz of this demo issue he told me? dont do anything until i tell
you" — explicitly paused the LLM before any code change to confirm no
retraining would be required.

**LLM:** Confirmed the plan would use a brand-new, in-memory `QLearningAgent`
for the demo, never touching the saved `qtable_2x2.pkl`. Waited for explicit
go-ahead.

**Me:** "yes" — proceed.

**LLM:** Created `demo_learning.py`, ran it, and showed the actual before/after
console output (Q-values all 0 before training, `U'` clearly highest at
+72.5 after 2,000 episodes) before declaring it done.

**Me:** "what about documentation fix and llm?"

**LLM:** Updated `README.md` (architecture tree, project layout, "Notable
implementation details") to match the now-true claim that the UI has no
business logic, and built this `llm/` folder.
