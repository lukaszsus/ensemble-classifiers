import os
import pandas as pd
import numpy as np

from ensemble.plotter import OUTCOME_PATH, col_names_to_groupby, datestamp
from ensemble.utils import create_dir_if_not_exists, is_number, is_int


def make_metrics_for_all_datasets_average_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        _make_metrics_average_folds_one_dataset(ds_data, ds_name)


def _make_metrics_average_folds_one_dataset(df, ds_name):
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    col_names = ['classifier', "param", "param value", 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']

    df = df[col_names]
    df = df.round({"acc_mean": 3, "prec_mean": 3, "rec_mean": 3, "f1_mean": 3})

    _save_df_to_csv(df, ds_name)
    _save_df_to_latex(df, ds_name)


def _save_df_to_csv(df: pd.DataFrame, sample_name: str):
    path = "table-{}-metrics.csv".format(sample_name)
    dir = os.path.join(OUTCOME_PATH, datestamp)
    dir = os.path.join(dir, 'tables')
    create_dir_if_not_exists(dir)
    path = os.path.join(dir, path)
    df.to_csv(path, index=False)


def _save_df_to_latex(df: pd.DataFrame, sample_name: str):
    path = "table-{}-metrics.txt".format(sample_name)
    dir = os.path.join(OUTCOME_PATH, datestamp)
    dir = os.path.join(dir, 'tables')
    create_dir_if_not_exists(dir)
    path = os.path.join(dir, path)
    with open(path, 'w') as file:
        df.to_latex(buf=file, index=False, float_format='%.3f', decimal=',')

##################################################################3


def make_metrics_for_all_params(df: pd.DataFrame):
    params = df["param"].unique()
    for param in params:
        df_param = df[df['param'] == param]
        if param != 'n_estimators':
            _make_metrics_average_folds_one_param(df_param, param)
        else:
            for ds in df_param['dataset'].unique():
                ds_df = df_param[df_param['dataset']==ds]
                _make_metrics_average_folds_one_param(ds_df, param, ds)


def _make_metrics_average_folds_one_param(df, param, ds_name = None):
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    col_names = ['dataset', 'classifier', param, 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']

    df[param] = df['param value']
    df = df[col_names]
    if is_int(df.iloc[0, 2]):
        df[param] = df[param].astype(int)
        df = df.sort_values(by=['dataset', 'classifier', param])
    elif is_number(df.iloc[0, 2]):
        df[param] = df[param].astype(np.float64)
        df = df.sort_values(by=['dataset', 'classifier', param])
    df = df.round({"acc_mean": 3, "prec_mean": 3, "rec_mean": 3, "f1_mean": 3})

    name = param
    if ds_name is not None:
        name += "-{}".format(ds_name)
    _save_df_to_csv(df, name)
    _save_df_to_latex(df, name)


if __name__ == '__main__':
    file_path = "../outcomes/{}/all.csv".format(datestamp)
    df = pd.read_csv(file_path)
    # make_metrics_for_all_datasets_average_folds(df)
    make_metrics_for_all_params(df)
