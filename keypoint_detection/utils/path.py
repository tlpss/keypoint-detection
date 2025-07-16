import os
from pathlib import Path


def get_artifact_dir_path() -> str:
    """Returns a standardized path for storing artifacts (checkpoints).
    Ensures consistency across processes in distributed training.
    """
    # Use a simple relative path that all processes can agree on
    path = Path("./artifacts")
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())

def get_wandb_log_dir_path() -> str:
    """Returns a standardized path for wandb logs.
    Ensures consistency across processes in distributed training.
    """
    # Use a simple relative path that all processes can agree on
    path = Path("./wandb")
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())


if __name__ == "__main__":
    print(get_artifact_dir_path())
