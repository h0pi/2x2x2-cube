from cube_agent.ml.i_cube_policy import ICubePolicy
from cube_agent.infrastructure import qtable_repository


class TrainingService:
    """
    Handles Q-table persistence: save and load.
    Bridges the policy (ML layer) with the repository (infrastructure layer).
    """

    def __init__(self, policy: ICubePolicy):
        self._policy = policy

    def save_qtable(self, path: str = None) -> None:
        data = self._policy.get_qtable()
        if path:
            qtable_repository.save(data, path)
        else:
            qtable_repository.save(data)

    def load_qtable(self, path: str = None) -> bool:
        """Load Q-table into policy. Returns True if a file was found."""
        data = qtable_repository.load(path) if path else qtable_repository.load()
        if not data:
            return False
        self._policy.set_qtable(data)
        return True

    def qtable_exists(self, path: str = None) -> bool:
        return qtable_repository.exists(path) if path else qtable_repository.exists()
