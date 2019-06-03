import pandas as pd

from sklearn import datasets


def load_iris():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    return data, target


def load_diabetes():
    diabetes = pd.read_csv('../data/pima_diabetes/diabetes.data', header=None)
    data = diabetes.iloc[:, 0:-1].values
    target = diabetes.iloc[:, -1].values
    return data, target


def load_glass():
    glass = pd.read_csv('../data/glass_identification/glass.data', header=None,
                        usecols=range(1,11))
    data = glass.iloc[:, 0:9].values
    target = glass.iloc[:, 9].values
    return data, target


def load_wine():
    wine = pd.read_csv('../data/wine/wine.data', header=None)
    data = wine.iloc[:, 1:].values
    target = wine.iloc[:, 0].values
    return data, target