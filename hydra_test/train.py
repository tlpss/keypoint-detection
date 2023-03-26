import hydra
import wandb
from example import Config, DatasetConfig, Model, ModelConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="dataset", name="default", node=DatasetConfig)
cs.store(group="model", name="default", node=ModelConfig)


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: Config):
    wandb.init(project=cfg.wandb_project, mode="offline" if cfg.wandb_offline else "online")
    # wandb.config.update(OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    # )
    print(cfg)
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    model = Model(cfg.model)  # noqa


if __name__ == "__main__":
    main()
