import os
from pathlib import Path


def get_artifact_dir_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "logging" / "artifacts"
    if not os.path.exists(path):
        path.mkdir(parents=True)
    return str(path)


def get_wandb_log_dir_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "logging" / "wandb"
    if not os.path.exists(path):
        path.mkdir(parents=True)
    return str(path)


if __name__ == "__main__":
    print(get_artifact_dir_path())
