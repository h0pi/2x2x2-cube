# LLM usage log

This folder documents how an LLM (Claude, via Claude Code) was used while building
and fixing this project. It is meant to be read together with the code — every
entry below points to a real file and a real change that I can explain in my own
words at the defense.

Contents:

- `prompts.md` — the key prompts I sent, grouped by the problem being solved.
- `before_after.md` — a table of concrete changes: file, what it looked like
  before, what it looks like after, and why.
- `conversation_log.md` — a condensed export of the actual back-and-forth for
  the two biggest debugging sessions (the orientation bug, and the
  architecture clean-up requested by the professor's feedback).

## How the LLM was used (summary)

1. **Bug diagnosis** — I described a symptom I could observe but not explain
   ("agent says solved, cube looks unsolved, only sometimes"). The LLM did not
   guess a fix immediately; it first added print-based diagnostics
   (`[CHECK] ...` lines in `cube_ui/main.py`) so the actual mismatch could be
   captured from a real run, then built an independent geometric simulation of
   the cube to reproduce the exact failing move sequence offline and test
   hypotheses against it (see `cube_agent/domain/cube_state.py`,
   `ORI_DELTA`).
2. **Targeted refactors from instructor feedback** — for each rubric point
   (atomic save, curriculum out of UI layer, learning demo), I had the LLM
   implement one change at a time, and I asked for verification after each
   one (e.g. a throwaway script proving the atomic save leaves no `.tmp`
   file, a script proving the curriculum logic still promotes/boosts
   identically after being moved).
3. **No blind retraining** — I explicitly required that none of the fixes
   touch or invalidate the existing trained `qtable_2x2.pkl` unless strictly
   necessary (the orientation fix was the one exception, because it changes
   the state encoding itself).
4. **Documentation kept in sync** — after each code change, the LLM updated
   `README.md` so the written description of the architecture matches what
   the code actually does (this was itself one of the professor's required
   fixes).
