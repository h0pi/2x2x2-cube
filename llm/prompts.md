# Key prompts sent to the LLM

Grouped by problem. Quoted prompts are verbatim where I have the exact text;
others are summarized from the session.

## 1. Diagnosing the "solved but visually wrong" bug

- *"I have an agent for solving 2x2x2 rubiks cube [...] please try to find the
  issue, it was not happening every time before this fix but now I feel like
  its happening less than before, find the issue, point it out, don't change
  anything in code"* — explicitly asked for diagnosis only, no blind fixes.
- *"lets start from the beginning try to find why it solves it but shoves
  unsolved state, maybe it is actually unsolved but just stops solving and
  says solved for some reason, scan the project carefully and find the
  issue"*
- After being shown console dumps of corner orientation matrices with
  non-identity rotations even though the agent reported "solved": *"yes"*
  (agreed to add a `[CHECK]`-style diagnostic comparing the rendered cube's
  geometry against the logical `Cube2x2State`).
- I rejected two earlier blind attempts at fixing the renderer itself
  (jitter removal, resetting the cube on "solved") with: *"revert all changes
  [...] I dont have issues like these before"* and *"revert this is so bad it
  just renders to solved state when its not solved I dont want this"* — this
  is why the final approach moved to fixing the underlying math
  (`ORI_DELTA`) instead of papering over it in the renderer.
- Follow-up after the first patch (which only fixed `L`) turned out
  incomplete: I pasted four more real `[CHECK]` failure logs from actual app
  runs and asked the LLM to find the full fix, not just the one case already
  covered.

## 2. UI/UX requests (unrelated to the bug, but in the same session)

- *"can you make window a bit bigger and another issue I have is that cube
  rotates when I hover with my mouse over it I want it to rotate only when I
  click and move mouse"*

## 3. Documentation

- *"can you now make a documentation (new notepad file) of my agent about my
  agent, I have to send this agent as a zip and that documentation to my
  professor."*
- *"can you give me architecture here so I can screenshot it and paste that
  image instead of this text"* — requested a diagram instead of an ASCII
  tree, then iterated twice on overlapping text/arrows in the diagram.

## 4. Acting on the professor's feedback

- Pasted the professor's full feedback (in Bosnian) and asked: *"can you
  separate this issues and get to them one by one but first tell me what are
  the fixes"* — asked for a plain-language breakdown before any code change.
- *"yes"* — confirmed go-ahead, starting with the atomic save fix (smallest,
  safest) before the bigger curriculum refactor.
- *"what about second issue? how to do that"* and *"do i need to retrain my
  agent bcz of this demo issue he told me? dont do anything until i tell
  you"* — I explicitly checked whether the learning-demo fix would force a
  retrain before allowing any code change; the LLM confirmed it would use a
  throwaway in-memory agent instead.
- *"what about documentation fix and llm?"* — final round, syncing
  `README.md` and building this `llm/` folder.

## Pattern across the whole session

I consistently asked the LLM to:
1. explain *what* it would change before changing it,
2. verify each change with a runnable check (a script, a re-run of the
   failing scenario, or sample output) rather than asserting it works, and
3. avoid touching the trained Q-table unless a fix made that unavoidable
   (only the orientation fix did).
