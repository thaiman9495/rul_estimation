import hydra

from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import List


@dataclass
class LstmModel:
    input_size: int = 24
    output_size_1: int = 64
    hidden_size_1: List[int] = field(default_factory=lambda: [128, 64])
    hidden_size_2: List[int] = field(default_factory=lambda: [64, 32])
    hidden_size_lstm: int = 64
    num_layers_lstm: int = 1


@dataclass
class Config:
    mode: str = 'training'
    dataset_name: str = 'FD001'

    n_epochs: int = 1500
    batch_size: int = 4
    lr: float = 0.00025
    device: str = 'cuda:0'

    lstm_model: LstmModel = LstmModel()


cs = ConfigStore.instance()
cs.store(name="config_to_yaml", node=Config)


@hydra.main(version_base=None, config_name="config_to_yaml")
def dump_yaml(cfg: Config):
    path = Path.cwd().parents[1].joinpath('config/config.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        OmegaConf.save(cfg, f)


if __name__ == '__main__':
    dump_yaml()
