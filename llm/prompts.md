# Key prompts sent to the LLM

Grouped by problem. Quoted prompts are verbatim where I have the exact text;
others are summarized from the session.

## 0. Original requirements & initial implementation (earlier session)

The project started from a Clean Architecture requirement my professor gave
for a different (C#) reference project, an "AiAgents" spam classifier with
`AiAgents.Core` (generic `SoftwareAgent<TPercept,TAction,TResult,TExperience>`
abstractions), a `SpamAgent` shared/domain/application layer, and a thin
`SpamAgent.Web` host. I pasted that full requirement (in Bosnian) and asked
the LLM to translate the same rules to my own Python project:

- *"Eh ja imam ovaj project 2x2x2-cube i u njemu ima 2x2x2 rubikova kocka
  koja se pokaze u prozoru i napravi par random okretanja, ja sam pokusao
  napraviti agenta i istrenirati ga da zna sloziti tu 2x2 kocku [...] hocu da
  ti meni napravis tog agenta pa da ga ja istreniram [...] ne moze se
  trenirati u isto vrijeme dok traje visualisacija okretanja strana kocke pa
  bih ja htio to razdvojiti [...] daj mi najbolje savjete da ja to pregledam
  pa cu ti onda kasnije reci kad ces kreniti u implementaciju"* — i.e. asked
  for an analysis/plan first (mapping the professor's Sense→Think→Act→Learn
  and Domain/Application/Infrastructure/Web rules onto `agents_core/`,
  `cube_agent/`, `cube_ui/`), and only approved implementation afterward with
  *"ok spreman sam pocni sa implementacijom"*.
- *"please continue where you left off"* — used after the session ran long
  and got summarized, to resume implementation without losing the plan.
- *"can you tell me is it optimized as much as it can be because I left it to
  train over one day and night it got to 5th depth and around 60% it is so
  slow and it uses 18gb of ram its too much I think"* — this is what led to
  replacing the Python `dict` Q-table with the numpy float32 array (see
  README "Memory efficiency").
- *"right now it considers cube solved only if orange is on top and white in
  front etc, it doesnt matter color orientation when solved it has only to be
  solved can you fix that and also when scrambling make sure it doesnt make
  one move and then makes same move backwards [...] at the end of 5 move
  scramble I have 2 move scramble"* — led to the scramble-quality filter
  (`Cube2x2State.scramble()` rejecting same-face/opposite-face repeats) and
  fixing the solved-check to be orientation-of-the-whole-cube-independent.
- *"yes but act as a senior ai engineer and make sure to not overlook
  anything"* — asked for a careful, non-superficial review rather than a
  quick patch, before a non-trivial logic fix.
- *"ok and when I do that change do I need to retrain it again?"* — recurring
  question throughout the project; I always asked this before accepting a
  fix that touches state representation.
- *"delete q table for me"* / *"delete q table for me please"* — used when a
  fix genuinely required starting training over (state encoding changes),
  as opposed to later fixes where I explicitly avoided this.
- *"update readme accordingly to agent updates by now"* — first request to
  keep `README.md` in sync with the code, which became a recurring step
  after every later change too.

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

## 3. Acting on the professor's feedback

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
