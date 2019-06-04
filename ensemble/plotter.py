import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from ensemble.utils import create_dir_if_not_exists, is_number

OUTCOME_PATH = '../outcomes/'
metrics_names = ['acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']
col_names_to_groupby = ["dataset", "classifier", "param", "param value"]
datestamp = '2019-06-03-19-39'


def plot_metrics_for_all_datasets_average_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        plot_for_all_params(ds_data, ds_name)


def plot_for_all_params(df: pd.DataFrame, ds_name):
    params = df["param"].unique()
    for param in params:
        if param == "params":
            continue
        plot_dependency_on_param(df, ds_name, param)


# def __multiplicate_basic_classifier_rows(df: pd.DataFrame):
#     values = df["param value"].unique()
#     basic_row = df[df["classifier"] == "DecisionTree"]
#     for i in range(len(values)):
#         row = basic_row.copy()
#         row["param"] = "n_estimators"
#         row["param value"] = values[i]
#         df.append(row, ignore_index=True)


def plot_dependency_on_param(df: pd.DataFrame, ds_name, param_name):
    df = df[df['param']==param_name]
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    if is_number(df.iloc[0, 3]):
        plot_fun = _plot_dependency_on_numerical_param
    else:
        plot_fun = _plot_dependency_on_binary_param

    for metric_name in metrics_names:
        if len(df["classifier"].unique()) > 1:
            plot_fun(df, ds_name, param_name, metric_name, hue='classifier')
        else:
            plot_fun(df, ds_name, param_name, metric_name)


def _plot_dependency_on_numerical_param(df: pd.DataFrame, ds_name, param_name, metric_name, hue=None):
    plt.clf()
    fig, ax = plt.subplots()
    df["param value"] = df["param value"].astype(np.float64)
    if hue is not None:
        ax = sns.lineplot(x="param value", y=metric_name, data=df, hue=hue, marker="o", markers=True, dashes=True)
    else:
        ax = sns.lineplot(x="param value", y=metric_name, data=df, marker="o", markers=True, dashes=True)
    for i in range(len(ax.lines)):
        ax.lines[i].set_linestyle("--")

    ax.set_title("Miara {} w zaleznosci od {} dla zbioru {}.".format(metric_name, param_name, ds_name))
    path = os.path.join(OUTCOME_PATH, datestamp)
    path = os.path.join(path, "plots")
    create_dir_if_not_exists(path)
    path_pdf = os.path.join(path, "{}-{}-{}.pdf".format(ds_name, param_name, metric_name))
    path_png = os.path.join(path, "{}-{}-{}.png".format(ds_name, param_name, metric_name))
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight')


def _plot_dependency_on_binary_param(df: pd.DataFrame, ds_name, param_name, metric_name, hue=None):
    plt.clf()
    fig, ax = plt.subplots()
    if hue is not None:
        ax = sns.barplot(x="param value", y=metric_name, data=df, hue=hue)
    else:
        ax = sns.barplot(x="param value", y=metric_name, data=df)

    ax.set_title("Miara {} w zaleznosci od {} dla zbioru {}.".format(metric_name, param_name, ds_name))
    path = os.path.join(OUTCOME_PATH, datestamp)
    path = os.path.join(path, "plots")
    create_dir_if_not_exists(path)
    path_pdf = os.path.join(path, "{}-{}-{}.pdf".format(ds_name, param_name, metric_name))
    path_png = os.path.join(path, "{}-{}-{}.png".format(ds_name, param_name, metric_name))
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight')


if __name__ == '__main__':
    file_path = "../outcomes/{}/all.csv".format(datestamp)
    df = pd.read_csv(file_path, index_col=False)
    plot_metrics_for_all_datasets_average_folds(df)
