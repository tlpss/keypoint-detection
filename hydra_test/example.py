import dataclasses

from omegaconf import MISSING


@dataclasses.dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING


@dataclasses.dataclass
class ModelConfig:
    name: str = "convnext"
    pretrained: bool = True
    channels: int = 4
    n_resnet_blocks: int = 3
    s: int = 3


@dataclasses.dataclass
class Config:
    wandb_project: str
    wandb_offline: bool = False
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)


class Model:
    def __init__(self, config=ModelConfig) -> None:
        assert isinstance(config, ModelConfig)
        self._config = config
