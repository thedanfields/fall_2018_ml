from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from cars_data import CarData
import seaborn as sns
import time
import pandas as pd
import matplotlib.pyplot as plt
import collections

def get_train_test_split(df_features, df_target):
    return train_test_split(df_features, df_target, test_size=0.33, random_state=42)


def get_neural_network_details(name, score, n_network, fit_time, score_time, print_details=True):

    index = str(len(n_network.hidden_layer_sizes)) + "\n(" + ".".join(map(str, n_network.hidden_layer_sizes)) + ")"

    if print_details:
        print 'Results for {}\n Index: {}\n Accuracy: {}\n Fit Time: {}\n Score Time: {}\n' \
            .format(name, index, score, fit_time, score_time)

    return pd.Series(dict(name=name, index=index, score=score, fit_time=fit_time, score_time=score_time))


def neural_network_experiment(ex_data, number_of_layers, number_of_neurons, max_iterations=500, print_details=True):
    x_train, y_train, x_test, y_test = ex_data['x_train'], ex_data['y_train'], ex_data['x_test'], ex_data['y_test']

    hidden_layers = []
    for layer in range(1, number_of_layers + 1):
        hidden_layers.append(number_of_neurons)

    mlp = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers), max_iter=max_iterations)
    start = time.time()
    mlp.fit(x_train, y_train)
    end = time.time()
    fit_time = (end - start)

    start = time.time()
    score = mlp.score(x_test, y_test)
    end = time.time()
    score_time = (end - start)

    details = get_neural_network_details("Neural Network w/ {} layers".format(number_of_layers),
                                         score, mlp, fit_time=fit_time, score_time=score_time,
                                         print_details=print_details)

    return details


def run_neural_network_experiment(name, df_features, series_target, min_number_of_layers, max_number_of_layers):
    x_train, x_test, y_train, y_test = get_train_test_split(df_features, series_target)
    data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    number_of_neurons = car_data.features.count(axis=1)[0]

    experiments = []

    layer_range = range(min_number_of_layers, max_number_of_layers + 1)

    for layer in layer_range:
        experiments.append(neural_network_experiment(data, layer, number_of_neurons))

    df_experiments = pd.DataFrame(experiments)

    df_experiments.to_csv('./report_artifacts/data/neural_network/' + name.lower().replace(' ', '_') + '.csv',
                          index=False, index_label='index')

    ax = sns.barplot(x="index", y="score", data=df_experiments)
    ax.set_title(name)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("layer")

    plt.ylim(df_experiments.score.min() - (df_experiments.score.min() * .01), df_experiments.score.max())

    fig = ax.get_figure()
    fig.savefig('./report_artifacts/figures/neural_network/' + name.lower().replace(' ', '_') + '.png')

    plt.show()


def run_experiments(car_data):
    min_number_of_layers = 2
    max_number_of_layers = 5

    Experiment = collections.namedtuple('Experiment', 'name target')

    experiments = [Experiment(name="Neural Network Classification of Very Good Cars", target=car_data.target_classification_very_good),
                   Experiment(name="Neural Network Classification Classification of Good Cars", target=car_data.target_classification_good),
                   Experiment(name="Neural Network Classification Classification of Acceptable", target=car_data.target_classification_acceptable),
                   Experiment(name="Neural Network Classification Classification of UnAcceptable", target=car_data.target_classification_unacceptable),
                   Experiment(name="Neural Network Classification Classification of Cars", target=car_data.target_classification)]

    for e in experiments:
        run_neural_network_experiment(e.name, car_data.features, e.target, min_number_of_layers, max_number_of_layers)

    # run_neural_network_experiment("Classification of Very Good Cars",
    #                               car_data.features,
    #                               car_data.target_classification_very_good,
    #                               min_number_of_layers, max_number_of_layers)

car_data = CarData()

run_experiments(car_data)
