from dataclasses import dataclass


@dataclass
class SolvingTickResult:
    """Result of one Sense->Think->Act cycle in the solving agent."""
    action_str: str
    action_idx: int
    is_solved: bool
    step_count: int
    done: bool      # True when solved or max_steps reached


@dataclass
class TrainingTickResult:
    """Result of one Sense->Think->Act->Learn cycle in the training agent."""
    episode_done: bool
    solved: bool
    epsilon: float
    episode_count: int
