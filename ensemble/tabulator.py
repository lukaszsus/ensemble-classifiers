import os
import pandas as pd

from .plotter import col_names_to_groupby, OUTCOME_PATH


def make_metrics_for_all_datasets_average_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        _make_metrics_average_folds_one_dataset(ds_data, ds_name)


def _table_for_all_params(df: pd.DataFrame, ds_name):
    params = df["param"].unique()
    for param in params:
        _make_metrics_average_folds_one_dataset(df, ds_name, param)


def _make_metrics_average_folds_one_dataset(df, param):
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    if param == 'n_estimators':
        col_names = ["dataset", 'classifier', param, 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']
    else:
        col_names = ["dataset", param, 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']

    df = df[col_names]
    df = df.round({"acc_mean": 3, "prec_mean": 3, "rec_mean": 3, "f1_mean": 3})

    _save_df_to_csv(df, param)
    _save_df_to_latex(df, param)


def _save_df_to_csv(df: pd.DataFrame, param: str):
    path = "table-{}-metrics.csv".format(param)
    path = os.path.join(OUTCOME_PATH, path)
    df.to_csv(path, index=False)


def _save_df_to_latex(df: pd.DataFrame, param: str):
    path = "table-{}-metrics.txt".format(param)
    path = os.path.join(OUTCOME_PATH, path)
    with open(path, 'a') as file:
        df.to_latex(buf=file, index=False, float_format='%.3f', decimal=',')


# def get_max_f1_for_every_k(df):
#     datasets = df["dataset"].unique()
#     ks = df["n-k"].unique()
#     df = df.groupby(col_names_to_groupby, as_index=False).mean()
#     for ds_name in datasets:
#         df_ds = df[df["dataset"]==ds_name]
#         iterator = 0
#         multiplier = 6  # 6 rows for every k
#
#         print("For {}".format(ds_name))
#         for k in ks:
#             df_k=df_ds[df_ds["n-k"]==k]
#             df_k=df_k[["n-k", "f1_mean"]]
#             index = df_k.idxmax(0)["f1_mean"]
#             #print("{}".format(str(index+iterator*multiplier)))
#             print()
#             row = df.iloc[index,:]
#             print(row)
#             iterator += 1


if __name__ == '__main__':
    file_path = "../outcomes/2019-05-14-16-00/all.csv"
    df = pd.read_csv(file_path)
    make_metrics_for_all_datasets_average_folds(df)
    # get_max_f1_for_every_k(df)
