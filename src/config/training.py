import hydra

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import List


def make_list(list_in):
    return [i for i in list_in]


@dataclass
class GeneralTrainingConfig:
    n_epochs: int = 1500
    batch_size: int = 4
    lr: float = 0.00025
    device: str = 'cuda:0'


@dataclass
class ModelConfig:
    input_size: int = 24
    output_size_1: int = 64
    hidden_size_1: List[int] = field(default_factory=lambda: [128, 64])
    hidden_size_2: List[int] = field(default_factory=lambda: [64, 32])
    hidden_size_lstm: int = 64
    num_layers_lstm: int = 1


@dataclass
class TrainingConfig:
    general: GeneralTrainingConfig = GeneralTrainingConfig()
    model: ModelConfig = ModelConfig()


cs = ConfigStore.instance()
cs.store(name="training_config", node=TrainingConfig)


@hydra.main(version_base=None, config_name="training_config")
def dump_yaml(cfg: TrainingConfig):
    with open("../../config/training.yaml", "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == '__main__':
    dump_yaml()
