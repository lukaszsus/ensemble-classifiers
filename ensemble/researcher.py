import datetime
import re
import warnings

from functools import partial
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier

from .plotter import OUTCOME_PATH
from .utils import *
from .dsloader import *

warnings.filterwarnings('ignore')


class Researcher:
    NUM_TESTS_PER_EXAMPLE = 10
    NUM_FOLDS_MIN = 2
    NUM_FOLDS_MAX = 9
    METRICS_COLUMN_LIST = ["dataset", "classifier", "param", "param value", "n-folds",
                           "acc_mean", "prec_mean", "rec_mean", "f1_mean"]

    def __init__(self, standarization=True):
        self._accuracies = None
        self._precisions = None
        self._recalls = None
        self._f1_scores = None

        self._splitter = None
        self._metrics: pd.DataFrame = pd.DataFrame(columns=self.METRICS_COLUMN_LIST)
        self._outcomes_dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        self._standarization = standarization
        self._scaler = preprocessing.StandardScaler()

        # params for research
        self._loader = None
        self._n_folds = None
        self._k = None
        self._voting_index = None
        self._voting = None
        self._dist = None

        # auxiliary
        self._data = None
        self._target = None

    def research_for_datasets(self):
        dataset_loaders = self._load_datasets()
        for self._loader in dataset_loaders:
            self._research_ensembles()

    def _research_ensembles(self):
        self._clf_name = "DecisionTree"
        params = {"params": ["default"]}
        for key, val in params.items():
            self.research_param(key, val)

        self._clf_name = "Bagging"
        params = {"n_estimators": [1,5,10,15,20,25,50,100,200,300,400,500],
                 "max_samples": [0.2,0.5,0.9,1.0],
                 "max_features": [0.2,0.5,0.9,1.0],
                 "bootstrap": [True, False]}
        for key, val in params.items():
            self.research_param(key, val)

        self._clf_name = "RandomForest"
        params = {"n_estimators": [1, 5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500],
                  "max_features": [0.2, 0.5, 0.9, 1.0],      # , 'sqrt', 'log2'],
                  "bootstrap": [True, False]}
        for key, val in params.items():
            self.research_param(key, val)

        self._clf_name = "AdaBoost"
        params = {"n_estimators": [1, 5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500],
                  "learning_rate": [0.5, 1.0, 2.0],     # how weights of examples change
                  "algorithm": ['SAMME', 'SAMME.R']}
        for key, val in params.items():
            self.research_param(key, val)

    def research_param(self, param_name, param_values):
        for param_value in param_values:
            #print("Research for: {}, {}={}".format(self._clf_name, param_name, param_value))

            self._param_name = param_name
            self._param_value = param_value

            create_clf = self._make_create_clf_function(self._clf_name, param_name, param_value)
            self._clf = create_clf()
            self._do_research_for_folds()

    @staticmethod
    def _make_create_clf_function(clf_name, param_name, param_value):
        params = {param_name: param_value}
        if clf_name=="DecisionTree":
            return partial(DecisionTreeClassifier)
        elif clf_name=="Bagging":
            return partial(BaggingClassifier, base_estimator=DecisionTreeClassifier(), n_jobs=-1, **params)
        elif clf_name=="RandomForest":
            return partial(RandomForestClassifier, n_jobs=-1, **params)
        elif clf_name=="AdaBoost":
            return partial(AdaBoostClassifier, base_estimator=DecisionTreeClassifier(), **params)

    def _do_research_for_folds(self):
        self._data, self._target = self._loader()

        if self._standarization:
            self._data = self._scaler.fit_transform(self._data)

        for self._n_folds in range(self.NUM_FOLDS_MIN, self.NUM_FOLDS_MAX + 1):
            self._do_research_for_n_samples()

    def _do_research_for_n_samples(self):
        self._splitter = StratifiedKFold(n_splits=self._n_folds, shuffle=True)
        self._target = translate_class_labels(self._target)
        self._make_n_samples()
        self._metrics_summary()
        self._save_to_file()

    def _load_datasets(self):
        datasets_loader = list()
        # datasets_loader.append(load_iris)
        datasets_loader.append(load_diabetes)
        datasets_loader.append(load_glass)
        datasets_loader.append(load_wine)
        return datasets_loader

    def _do_crossval(self):
        self._splitter = StratifiedKFold(n_splits=self._n_folds, shuffle=True)
        split_set_generator = self._splitter.split(self._data, self._target)

        # trainning and testing
        y_pred = list()
        y_true = list()

        for train_indices, test_indices in split_set_generator:
            X_train = self._data[train_indices]
            Y_train = self._target[train_indices]

            self._clf.fit(X_train, Y_train)

            y_pred.extend(self._clf.predict(self._data[test_indices]))
            y_true.extend(self._target[test_indices])

        confusion = metrics.confusion_matrix(y_true, y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average=None)
        recall = metrics.recall_score(y_true, y_pred, average=None)
        f1_score = metrics.f1_score(y_true, y_pred, average=None)

        return {"confusion": confusion, "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1_score": f1_score}

    def _make_n_samples(self):
        self._accuracies = list()
        self._precisions = list()
        self._recalls = list()
        self._f1_scores = list()

        for i in range(self.NUM_TESTS_PER_EXAMPLE):
            metrics = self._do_crossval()

            self._accuracies.append(metrics["accuracy"])
            self._precisions.append(metrics["precision"])
            self._recalls.append(metrics["recall"])
            self._f1_scores.append(metrics["f1_score"])

        self._accuracies = np.asarray(self._accuracies)
        self._precisions = np.asarray(self._precisions)
        self._recalls = np.asarray(self._recalls)
        self._f1_scores = np.asarray(self._f1_scores)

    def _metrics_summary(self):
        mean_acc = np.mean(self._accuracies)
        mean_prec = np.mean(self._precisions)
        mean_rec = np.mean(self._recalls)
        mean_f1 = np.mean(self._f1_scores)

        record = pd.DataFrame([[self.__get_name_from_loader(),
                                self._clf_name,
                                self._param_name,
                                self._param_value,
                                self._n_folds,
                                mean_acc,
                                mean_prec,
                                mean_rec,
                                mean_f1]], columns=self.METRICS_COLUMN_LIST)
        if self._metrics.empty:
            self._metrics = record
        else:
            self._metrics = pd.concat([self._metrics, record], ignore_index=True)

        print("Dataset: {0}".format(self.__get_name_from_loader()))
        print("Classifier: {0}".format(self._clf_name))
        print("Param: {0}".format(self._param_name))
        print("Param-value: {0}".format(self._param_value))
        print("N-folds: {0}".format(self._n_folds))
        print("Accuracy:\nmean: {0}".format(mean_acc))
        print("Precision:\nmean: {0}".format(mean_prec))
        print("Recall:\nmean: {0}".format(mean_rec))
        print("F1 score:\nmean: {0}".format(mean_f1))
        print()

    def _save_to_file(self):
        file_name = "{}.csv".format(self.__get_name_from_loader())
        path = os.path.join(OUTCOME_PATH, self._outcomes_dir_name)
        create_dir_if_not_exists(path)
        path = os.path.join(path, file_name)
        self._metrics.to_csv(path, index=False)

    def __get_name_from_loader(self):
        name = self._loader.__name__
        name = re.search("_.*", name)
        name = name.group(0)[1:]
        return name
