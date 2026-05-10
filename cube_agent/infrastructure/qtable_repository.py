import os
import pickle

# Default path: project root / qtable_2x2.pkl
# __file__ is cube_agent/infrastructure/qtable_repository.py
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
DEFAULT_PATH = os.path.join(_PROJECT_ROOT, 'qtable_2x2.pkl')


def save(qtable: dict, path: str = DEFAULT_PATH) -> None:
    """Persist Q-table dict to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(qtable, f)
    print(f"[Repository] Q-table saved to {path}  ({len(qtable)} entries)")


def load(path: str = DEFAULT_PATH) -> dict:
    """Load Q-table dict from a pickle file. Returns {} if file not found or corrupt."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"[Repository] Q-table loaded from {path}  ({len(data)} entries)")
        return data
    except Exception as e:
        print(f"[Repository] WARNING: could not load Q-table ({e}) — starting fresh.")
        os.remove(path)
        return {}


def exists(path: str = DEFAULT_PATH) -> bool:
    return os.path.exists(path)
