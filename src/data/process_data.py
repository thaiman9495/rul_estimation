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


def process_data(path_train_data: Path, path_test_data: Path, max_rul: float):
    train_df = pd.read_csv(path_train_data, sep='\s+', header=None, names=cmapss_column.name_list)
    test_df = pd.read_csv(path_test_data, sep='\s+', header=None, names=cmapss_column.name_list)

    # Apply max-min normalization for training data

    # Add RUL to training data
    train_df = add_rul(train_df, max_rul)

    # Convert to list of torch tensors
    train_df = train_df.groupby(cmapss_column.engine_id)

    x = torch.Tensor(train_df.get_group(1).iloc[:, 2:-1].to_numpy())
    y = torch.Tensor(train_df.get_group(1).iloc[:, -1].to_numpy())

    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    print('Thai man is very handsome!')
