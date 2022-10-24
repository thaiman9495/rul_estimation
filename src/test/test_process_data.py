import pandas as pd

from pathlib import Path
from src.data.process_data import cmapss_column, add_rul


if __name__ == '__main__':
    # Choose your tests
    # 0: add_rul()
    # 1: normalize_data()
    test = 1

    # Configure paths
    path_parent = Path.cwd().parents[1]  # ../rul_estimation
    path_data = path_parent.joinpath('data/raw/cmapss/train_FD001.txt')

    # Load raw data
    raw_data = pd.read_csv(path_data, sep='\s+', header=None, names=cmapss_column.name_list)
    print("Raw data")
    print(raw_data.head())

    if test == 0:
        print("Data with added RUL column")
        data_with_rul = add_rul(raw_data, 130, piecewise_linear_rul=True)
        print(data_with_rul)

    if test == 1:
        pass

    if test == 2:
        pass
