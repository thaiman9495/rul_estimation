import torch
import hydra

from pathlib import Path
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from hydra.core.config_store import ConfigStore
from utils.mini_batch import generate_list_of_mini_batches
from model.lstm_model import RluModel
from config.config_hydra import Config


cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)


def train(params: Config):
    data_name = params.dataset_name  # Name of considering dataset that can be change by command line
    tensor_board_writer = SummaryWriter(log_dir=f'../runs/{data_name}', flush_secs=30)

    # Load processed data
    path_parent = Path.cwd().parent
    train_data = torch.load(path_parent.joinpath(f'data/processed/cmapss/train_{data_name}.pt'))
    val_data = torch.load(path_parent.joinpath(f'data/processed/cmapss/val_{data_name}.pt'))

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
        loss_train = 0.0

        # Update RUL model
        for batch in mini_batches:
            loss_batch = torch.tensor(0.0).to(params.device)
            n_loss_points = 0

            for feature_sq, rul_target_sq in batch:
                # Put feature_sq and rul_target_sq to training device (cpu or gpu)
                feature_sq = feature_sq.to(params.device)
                rul_target_sq = rul_target_sq.to(params.device)

                # Forward part
                rul_estimated_sq = rlu_model(feature_sq)
                loss_batch += torch.sum((rul_estimated_sq - rul_target_sq) ** 2)
                n_loss_points += len(feature_sq)

            optimizer.zero_grad()
            loss_batch = torch.sqrt(loss_batch / n_loss_points)  # Root-mean-square loss
            loss_batch.backward()                                # Backward part
            optimizer.step()                                     # Update training model

            with torch.no_grad():
                loss_train += loss_batch

        # Validate trained RUL model after every training epoch
        loss_validation = torch.tensor(0.0).to(params.device)
        n_loss_points_val = 0        # Count the numer of points in root-mean-square loss
        rlu_model.eval()             # Put RUL model in validation mode

        for feature_sq, rul_target_sq in val_data:
            feature_sq = feature_sq.to(params.device)
            rul_target_sq = rul_target_sq.to(params.device)

            rul_estimated_sq = rlu_model(feature_sq)
            loss_validation += torch.sum((rul_estimated_sq - rul_target_sq) ** 2)
            n_loss_points_val += len(feature_sq)

        loss_validation = torch.sqrt(loss_validation / n_loss_points_val)
        rlu_model.train()  # Put RUL model back to training mode

        # Log training and validation loss
        print(f'Epoch {epoch} -> training loss: {loss_train: .4f}, validation loss: {loss_validation: 4f}')

        tensor_board_writer.add_scalars(main_tag=f'log_{data_name}',
                                        tag_scalar_dict={'training_loss': loss_train,
                                                         'validation_loss': loss_validation},
                                        global_step=epoch)

    tensor_board_writer.close()


def test():
    # test_data = torch.load(path_parent.joinpath(f'data/processed/cmapss/test_{data_name}.pt'))
    pass


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def main(params: Config):
    if params.mode == 'training':
        train(params)
    else:
        test()


if __name__ == '__main__':
    main()



