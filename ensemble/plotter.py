import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

OUTCOME_PATH = '../outcomes/'
metrics_names = ['acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']
col_names_to_groupby = ["dataset", "classifier", "param", "param value"]


def plot_metrics_for_all_datasets_average_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        plot_for_all_params(ds_data, ds_name)


def plot_for_all_params(df: pd.DataFrame, ds_name):
    params = df["param"].unique()
    for param in params:
        plot_dependency_on_param(df, ds_name, param)


def plot_dependency_on_param(df: pd.DataFrame, ds_name, param_name):
    df = df[df['param name']==param_name]
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    if type(df.iloc[0, 3])==float or type(df.iloc[0, 3])==int:
        plot_fun = _plot_dependency_on_numerical_param
    else:
        plot_fun = _plot_dependency_on_binary_param

    for metric_name in metrics_names:
        if df["classifier"].unique() > 1:
            plot_fun(df, ds_name, param_name, metric_name, hue='classifier')
        else:
            plot_fun(df, ds_name, param_name, metric_name, hue='classifier')


def _plot_dependency_on_numerical_param(df: pd.DataFrame, ds_name, param_name, metric_name, hue=None):
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=param_name, y=metric_name, data=df, hue=hue, markers=True, dashes=True)
    for i in range(len(ax.lines)):
        if i % 2 == 0:
            ax.lines[i].set_linestyle("-.")
        else:
            ax.lines[i].set_linestyle("--")

    ax.set_title("Miara {} w zaleznosci od {} dla zbioru {}.".format(metric_name, param_name, ds_name))
    path = os.path.join(OUTCOME_PATH, "plots")
    path_pdf = os.path.join(path, "{}-{}.pdf".format(ds_name, metric_name))
    path_png = os.path.join(path, "{}-{}.png".format(ds_name, metric_name))
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight')


def _plot_dependency_on_binary_param(df: pd.DataFrame, ds_name, param_name, metric_name, hue=None):
    fig, ax = plt.subplots()
    ax = sns.barplot(x=param_name, y=metric_name, data=df, hue=hue)

    ax.set_title("Miara {} w zaleznosci od {} dla zbioru {}.".format(metric_name, param_name, ds_name))
    path = os.path.join(OUTCOME_PATH, "plots")
    path_pdf = os.path.join(path, "{}-{}.pdf".format(ds_name, metric_name))
    path_png = os.path.join(path, "{}-{}.png".format(ds_name, metric_name))
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight')


if __name__ == '__main__':
    file_path = "../outcomes/2019-05-16-17-36/all.csv"
    df = pd.read_csv(file_path)
    df.to_csv(file_path)
    plot_metrics_for_all_datasets_average_folds(df)
