import torch
import pandas as pd

from pathlib import Path


class CmapssColumn:
    def __init__(self):
        # The first and second column
        self.engine_id = 'engine_id'
        self.cycle = 'cycle'

        # 3 operating settings
        self.setting = [f'setting_{i}' for i in range(1, 4)]

        # 21 sensors
        self.sensor = [f'sensor_{i}' for i in range(1, 22)]

        # A list holds all columns' name of the CMAPSS dataset
        self.name_list = [self.engine_id] + [self.cycle] + self.setting + self.sensor

        # A list holds all columns' name of settings and sensors
        self.name_setting_sensor = self.setting + self.sensor


cmapss_column = CmapssColumn()


def normalize_data(data):
    """
    This functions aims to normalize data by using z-score normalization technique

    Args:
        data (pd.DataFrame) : raw CMAPSS training dataset

    Returns:
        pd.DataFrame: normalized data

    """

    normalized_data = data.copy()

    for colum_name in cmapss_column.name_setting_sensor:
        mean = data[colum_name].mean()
        std = data[colum_name].std()

        if std != 0:
            normalized_data[colum_name] = (data[colum_name] - mean) / std
        else:
            normalized_data[colum_name] = data[colum_name] - mean

    return normalized_data


def add_rul(data, max_rul, piecewise_linear_rul=True):
    """
    This function is to add RUL to CMAPSS training dataset
    Args:
        data (pd.DataFrame): CMAPSS training dataset
        max_rul (float): upper limit of RUL
        piecewise_linear_rul (bool): linear or piecewise linear RLU function
    Returns:
         pd.DataFrame: A processed DataFrame with a added RUL column
    """

    # Get the total number of operating cycles for each engine
    max_cycles = data.groupby(cmapss_column.engine_id)[cmapss_column.cycle].max()

    # Add raw RLU to data
    data_ = data.merge(max_cycles.to_frame(name='max_cycle'), left_on=cmapss_column.engine_id, right_index=True)
    rlu = data_['max_cycle'] - data_['cycle']
    processed_data = data_.drop(columns=['max_cycle'], axis=1)

    # Linear degradation model
    processed_data['rul'] = rlu

    # Piecewise linear degradation model
    if piecewise_linear_rul:
        processed_data.loc[processed_data['rul'] > max_rul, 'rul'] = max_rul

    return processed_data


def process_train_data(data_name: str, max_rul: float):
    path_parent = Path.cwd().parents[1]  # ../rul_estimation
    path_data = path_parent.joinpath('data/raw/cmapss/' + 'train_' + data_name + '.txt')
    data = pd.read_csv(path_data, sep='\s+', header=None, names=cmapss_column.name_list)

    # Apply z-score normalization for training data
    data = normalize_data(data)

    # Add RUL to training data
    data = add_rul(data, max_rul)

    # Convert to list of torch tensors
    processed_data = []
    data = data.groupby(cmapss_column.engine_id)
    n_sq = data.ngroups

    for i in range(1, n_sq + 1):
        # feature_sq has a shape of (T x M) in which T is sequence length and T is the number of sensor values
        # target_sq have shape of (T x 1)
        feature_sq = torch.FloatTensor(data.get_group(i).iloc[:, 2:-1].to_numpy())
        target_sq = torch.unsqueeze(torch.FloatTensor(data.get_group(i).iloc[:, -1].to_numpy()), 1)

        processed_data.append((feature_sq, target_sq))

    # Split data into training (75%) and validation (25%) set
    idx = int(0.75 * n_sq)
    processed_train_data = processed_data[0: idx]
    processed_val_data = processed_data[idx:]

    # Save the processed data
    torch.save(processed_train_data, path_parent.joinpath('data/processed/cmapss/' + 'train_' + data_name + '.pt'))
    torch.save(processed_val_data, path_parent.joinpath('data/processed/cmapss/' + 'val_' + data_name + '.pt'))


def process_test_data(data_name: str):
    path_parent = Path.cwd().parents[1]  # ../rul_estimation
    path_feature_data = path_parent.joinpath('data/raw/cmapss/' + 'test_' + data_name + '.txt')
    path_target_data = path_parent.joinpath('data/raw/cmapss/' + 'RUL_' + data_name + '.txt')

    feature_data = pd.read_csv(path_feature_data, sep='\s+', header=None, names=cmapss_column.name_list)
    target_data = pd.read_csv(path_target_data, sep='\s+', header=None)

    # Apply z-score normalization to feature data
    data = normalize_data(feature_data)

    # Convert to list of torch tensors
    processed_data = []
    data = feature_data.groupby(cmapss_column.engine_id)
    n_sq = data.ngroups

    for i in range(1, n_sq + 1):
        # feature_sq has a shape of (T x M) in which T is sequence length and T is the number of sensor values
        # target have shape of (1 x 1)
        feature_sq = torch.FloatTensor(data.get_group(i).iloc[:, 2:-1].to_numpy())
        target = torch.unsqueeze(torch.FloatTensor(target_data.iloc[i-1, :].to_numpy()), 1)

        processed_data.append((feature_sq, target))

    # Save processed data
    torch.save(processed_data, path_parent.joinpath('data/processed/cmapss/' + 'test_' + data_name + '.pt'))


if __name__ == '__main__':
    # Names of sub-dataset of CMAPSS dataset
    data_names = ['FD001', 'FD002', 'FD003', 'FD004']

    # Process data
    for data_name in data_names:
        process_train_data(data_name, 130)
        process_test_data(data_name)
