import collections
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from cars_data import CarData


def get_train_test_split(df_features, df_target):
    return train_test_split(df_features, df_target, test_size=0.33, random_state=42)


def get_knn_details(name, score, learner, fit_time, score_time, print_details=True):
    index = learner.n_neighbors

    if print_details:
        print 'Results for {}\n Index: {}\n Accuracy: {}\n Fit Time: {}\n Score Time: {}\n n_neighbors: {}\n'.format(
            name, index, score, fit_time, score_time, learner.n_neighbors)

    return pd.Series(dict(name=name, index=index, score=score, fit_time=fit_time, score_time=score_time,
                          n_neighbors=learner.n_neighbors))


def knn_experiment(ex_data, n_neighbors=2, print_details=True):
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

    details = get_knn_details("KNN", score, learner, fit_time=fit_time, score_time=score_time,
                              print_details=print_details)

    return details


def run_knn_experiment(name, df_features, series_target, min_k, max_k):
    x_train, x_test, y_train, y_test = get_train_test_split(df_features, series_target)
    data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    experiments = []

    k_range = range(min_k, max_k + 1)

    for k in k_range:
        experiments.append(knn_experiment(data, k))

    df_experiments = pd.DataFrame(experiments)

    df_experiments.to_csv('./report_artifacts/data/knn/' + name.lower().replace(' ', '_') + '.csv',
                          index=False, index_label='index')

    ax = sns.barplot(x="index", y="score", data=df_experiments)
    ax.set_title(name)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("k value")

    plt.ylim(df_experiments.score.min() - (df_experiments.score.min() * .01), df_experiments.score.max())

    fig = ax.get_figure()
    fig.savefig('./report_artifacts/figures/knn/' + name.lower().replace(' ', '_') + '.png')

    plt.show()


def run_experiment(car_data):
    sns.set(style="whitegrid")
    # sns.set(font_scale=.75)
    Experiment = collections.namedtuple('Experiment', 'name target')

    min_k = 1
    max_k = 10

    experiments = [Experiment(name="Very Good Cars", target=car_data.target_classification_very_good),
                   Experiment(name="Good Cars", target=car_data.target_classification_good),
                   Experiment(name="Acceptable Cars", target=car_data.target_classification_acceptable),
                   Experiment(name="UnAcceptable Cars", target=car_data.target_classification_unacceptable),
                   Experiment(name="Cars", target=car_data.target_classification)
                   ]

    learner = "K-NN classification of "

    for e in experiments:
        run_knn_experiment("{} {}".format(learner, e.name), car_data.features, e.target, min_k, max_k)


car_data = CarData()
run_experiment(car_data)
