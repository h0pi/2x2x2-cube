"""
Demonstration that the agent's Learn step (Bellman update) actually changes
its behavior, in isolation from the UI/3D visualization.

Uses a FRESH, in-memory QLearningAgent — it never loads or saves
qtable_2x2.pkl, so running this script does not affect (or require) the
project's existing trained Q-table.

Scenario:
  1. Fix one cube state: solved cube + a single 'U' move.
  2. BEFORE learning: print Q-values for that state and the greedy action
     the agent would pick (table is all zeros, so this is effectively
     undefined/random — there's nothing learned yet).
  3. Run real training episodes (TrainingAgentRunner: Sense -> Think -> Act
     -> Learn) at scramble depth 1, so the agent repeatedly encounters this
     exact state, receives real environment rewards, and updates its
     Q-table via update().
  4. AFTER learning: print Q-values for the SAME state again and the new
     greedy action. It should now consistently prefer "U'" (the only move
     that solves the cube from this state), with a clearly higher Q-value
     than the other 11 actions.

Run with:  python demo_learning.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cube_agent.domain.cube_state import Cube2x2State
from cube_agent.domain.actions import IDX_TO_ACTION
from cube_agent.ml.qlearning_agent import QLearningAgent
from cube_agent.application.runners.training_agent_runner import TrainingAgentRunner

N_TRAINING_EPISODES = 2_000


def show_state(policy: QLearningAgent, state_encoded: int, label: str) -> None:
    q_values = policy.get_action_values(state_encoded)
    greedy_idx = policy.choose_action_greedy(state_encoded)

    print(f"\n--- {label} ---")
    for idx, q in enumerate(q_values):
        marker = "  <== greedy choice" if idx == greedy_idx else ""
        print(f"  {IDX_TO_ACTION[idx]:>3}: Q = {q:+.4f}{marker}")
    print(f"Chosen action: {IDX_TO_ACTION[greedy_idx]}")


def main() -> None:
    # ---- the one fixed cube state used for both snapshots ----
    fixed_state = Cube2x2State.solved().apply("U")
    fixed_state_encoded = fixed_state.encode()
    print(f"Fixed demo state: solved cube after applying 'U' "
          f"(encoded = {fixed_state_encoded})")
    print("The only single move that solves this state is \"U'\".")

    # ---- fresh, untrained policy — does NOT touch qtable_2x2.pkl ----
    policy = QLearningAgent()

    show_state(policy, fixed_state_encoded, "BEFORE learning (untrained Q-table)")

    # ---- real training: Sense -> Think -> Act -> Learn, depth-1 scrambles ----
    runner = TrainingAgentRunner(policy=policy, scramble_len=1, max_steps=20)
    for _ in range(N_TRAINING_EPISODES):
        runner.step()

    show_state(policy, fixed_state_encoded, "AFTER learning (2,000 real training episodes)")

    print(
        "\nConclusion: identical input (same encoded cube state) produces a "
        "different decision after the agent's update() has been called with "
        "real environment rewards — this is the Learn step changing behavior."
    )


if __name__ == "__main__":
    main()
