import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split


class KnnLearner:

    def __init__(self, figure_path, csv_path):
        self.figure_path = figure_path
        self.csv_path = csv_path

    def get_train_test_split(self, df_features, df_target):
        return train_test_split(df_features, df_target, test_size=0.33, random_state=42)

    def get_knn_details(self, name, score, learner, fit_time, score_time, print_details=True):
        index = learner.n_neighbors

        if print_details:
            print 'Results for {}\n Index: {}\n Accuracy: {}\n Fit Time: {}\n Score Time: {}\n n_neighbors: {}\n'.format(
                name, index, score, fit_time, score_time, learner.n_neighbors)

        return pd.Series(dict(name=name, index=index, score=score, fit_time=fit_time, score_time=score_time,
                              n_neighbors=learner.n_neighbors))

    def knn_experiment(self, ex_data, n_neighbors=2, print_details=True):
        x_train, y_train, x_test, y_test = ex_data['x_train'], ex_data['y_train'], ex_data['x_test'], ex_data['y_test']

        learner = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

        start = time.time()
        learner.fit(x_train, y_train)
        end = time.time()
        fit_time = (end - start)

        start = time.time()
        score = learner.score(x_test, y_test)
        end = time.time()
        score_time = (end - start)

        details = self.get_knn_details("KNN", score, learner, fit_time=fit_time, score_time=score_time,
                                  print_details=print_details)

        return details

    def run_knn_experiment(self, name, df_features, series_target, min_k, max_k):
        x_train, x_test, y_train, y_test = self.get_train_test_split(df_features, series_target)
        data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        experiments = []

        k_range = range(min_k, max_k + 1)

        for k in k_range:
            experiments.append(self.knn_experiment(data, k))

        df_experiments = pd.DataFrame(experiments)

        df_experiments.to_csv(self.csv_path + name.lower().replace(' ', '_') + '.csv',
                              index=False, index_label='index')

        ax = sns.barplot(x="index", y="score", data=df_experiments)
        ax.set_title(name)
        ax.set_ylabel("accuracy")
        ax.set_xlabel("k value")

        plt.ylim(df_experiments.score.min() - (df_experiments.score.min() * .01), df_experiments.score.max())

        fig = ax.get_figure()
        fig.savefig(self.figure_path + name.lower().replace(' ', '_') + '.png')

        plt.show()