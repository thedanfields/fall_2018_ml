import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split


class SVMLeaner:

    def __init__(self, figure_path, csv_path):
        self.figure_path = figure_path
        self.csv_path = csv_path


    def get_train_test_split(self, df_features, df_target):
        return train_test_split(df_features, df_target, test_size=0.33, random_state=42)

    def get_svm_details(self, name, score, svm, fit_time, score_time, print_details=True):

        index = svm.kernel + " (" + str(svm.degree) + ")"

        if print_details:
            print 'Results for {}\n Index: {}\n Accuracy: {}\n Fit Time: {}\n Score Time: {}\n Kernel: {}\n Degree: {}\n Gamma: {}\n'.format(name, index, score, fit_time, score_time, svm.kernel, svm.degree, svm.gamma)

        return pd.Series(dict(name=name, index=index, score=score, fit_time=fit_time, score_time=score_time,
                              kernel=svm.kernel, degree=svm.degree, gamma=svm.gamma))

    def svm_experiment(self, ex_data, degree, gamma='auto', kernel='rbf', print_details=True):
        x_train, y_train, x_test, y_test = ex_data['x_train'], ex_data['y_train'], ex_data['x_test'], ex_data['y_test']

        learner = svm.SVC(kernel=kernel, degree=degree, gamma=gamma)

        start = time.time()
        learner.fit(x_train, y_train)
        end = time.time()
        fit_time = (end - start)

        start = time.time()
        score = learner.score(x_test, y_test)
        end = time.time()
        score_time = (end - start)

        details = self.get_svm_details("SVM", score, learner, fit_time=fit_time, score_time=score_time,
                                  print_details=print_details)

        return details

    def run_SVM_experiment(self, name, df_features, series_target, min_degree, max_degree):
        x_train, x_test, y_train, y_test = self.get_train_test_split(df_features, series_target)
        data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        linear_experiments = []
        polynomial_experiments = []
        rbf_experiments = []
        sigmoid_experiments = []

        degree_range = range(min_degree, max_degree + 1)

        for degree in degree_range:
            linear_experiments.append(self.svm_experiment(data, degree, kernel='linear'))
            polynomial_experiments.append(self.svm_experiment(data, degree, kernel='poly', gamma=.66))
            rbf_experiments.append(self.svm_experiment(data, degree, kernel='rbf', gamma=.66))
            sigmoid_experiments.append(self.svm_experiment(data, degree, kernel='sigmoid'))

        df_linear = pd.DataFrame(linear_experiments)
        df_polynomial = pd.DataFrame(polynomial_experiments)
        df_rbf = pd.DataFrame(rbf_experiments)
        sigmoid_rbf = pd.DataFrame(sigmoid_experiments)

        df_experiments = pd.concat([df_linear, df_polynomial, df_rbf, sigmoid_rbf])

        df_experiments.to_csv(self.csv_path + name.lower().replace(' ', '_') + '.csv',
                              index=False, index_label='index')

        ax = sns.barplot(x="index", y="score", hue="kernel", data=df_experiments)
        ax.set_title(name)
        ax.set_ylabel("accuracy")
        ax.set_xlabel("kernel / degree")

        plt.ylim(df_experiments.score.min() - (df_experiments.score.min() * .01), df_experiments.score.max())

        fig = ax.get_figure()
        fig.savefig(self.figure_path + name.lower().replace(' ', '_') + '.png')

        plt.show()