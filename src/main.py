import torch
import hydra

from pathlib import Path
from torch.optim import Adam
from hydra.core.config_store import ConfigStore
from utils.mini_batch import generate_list_of_mini_batches
from model.lstm_model import RluModel
from config.config_hydra import Config


cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)


def train(params: Config):
    data_name = params.dataset_name

    # Load processed data
    path_parent = Path.cwd().parent
    train_data = torch.load(path_parent.joinpath(f'data/processed/cmapss/train_{data_name}.pt'))
    val_data = torch.load(path_parent.joinpath(f'data/processed/cmapss/val_{data_name}.pt'))
    # test_data = torch.load(path_parent.joinpath(f'data/processed/cmapss/test_{data_name}.pt'))

    # RUL model
    rlu_model = RluModel(params.lstm_model.input_size,
                         params.lstm_model.output_size_1,
                         params.lstm_model.hidden_size_1,
                         params.lstm_model.hidden_size_2,
                         params.lstm_model.hidden_size_lstm,
                         params.lstm_model.num_layers_lstm).to(params.device)

    # Optimizer
    optimizer = Adam(rlu_model.parameters(), lr=params.lr)

    # Training
    for epoch in range(params.n_epochs):
        mini_batches = generate_list_of_mini_batches(train_data, params.batch_size)
        loss_epoch = 0.0

        for batch in mini_batches:
            loss_batch = torch.tensor(0.0).to(params.device)

            for feature_sq, rul_target_sq in batch:
                # Put feature_sq and rul_target_sq to training device (cpu or gpu)
                feature_sq = feature_sq.to(params.device)
                rul_target_sq = rul_target_sq.to(params.device)

                # Forward part
                rul_estimated_sq = rlu_model(feature_sq)
                loss_batch += torch.sum((rul_estimated_sq - rul_target_sq) ** 2) / len(feature_sq)

            optimizer.zero_grad()

            # Root-mean-square loss
            loss_batch = torch.sqrt(loss_batch)

            # Backward part
            loss_batch.backward()

            # Update training model
            optimizer.step()

            loss_epoch += loss_batch

        print(f'Epoch {epoch}: {loss_epoch}')


def test():
    pass


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def main(params: Config):
    if params.mode == 'training':
        train(params)
    else:
        test()


if __name__ == '__main__':
    main()



