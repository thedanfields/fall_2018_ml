import collections

import seaborn as sns

from adult_data.adult_data import AdultData
from car_data.cars_data import CarData
from learners.decision_tree import DecisionTreeLeaner
from learners.knn import KnnLearner
from learners.neural_network import NeuralNetworkLearner
from learners.support_vector_machine import SVMLeaner

DecisionTreeExperiment = collections.namedtuple('DecisionTreeExperiment', 'name target prune_to max_tree_depth')
BoostedTreeExperiment = collections.namedtuple('BoostedTreeExperiment', 'name target max_tree_depth')
KnnExperiment = collections.namedtuple('KnnExperiment', 'name target min_k max_k')
NeuralNetworkExperiment = collections.namedtuple('NeuralNetworkExperiment', 'name target min_layers max_layers')
SVMExperiment = collections.namedtuple('SVMExperiment', 'name target min_degree max_degree')


def generate_decision_tree_experiments_for_car_data(data):
    prune_to = 10
    return [
        DecisionTreeExperiment(name="Very Good Cars", target=data.target_classification_very_good,
                               max_tree_depth=10, prune_to=prune_to),
        DecisionTreeExperiment(name="Good Cars", target=data.target_classification_good,
                               max_tree_depth=12, prune_to=prune_to),
        DecisionTreeExperiment(name="Acceptable Cars", target=data.target_classification_acceptable,
                               max_tree_depth=10, prune_to=prune_to),
        DecisionTreeExperiment(name="UnAcceptable Cars", target=data.target_classification_unacceptable,
                               max_tree_depth=10, prune_to=prune_to),
        DecisionTreeExperiment(name="Cars", target=data.target_classification,
                               max_tree_depth=13, prune_to=prune_to)
    ]


def generate_decision_tree_experiments_for_wage_data(data):
    prune_to = 10
    return [
        DecisionTreeExperiment(name="Wages 50k or less", target=data.target_classification_50_k_or_less,
                               max_tree_depth=10, prune_to=prune_to),
        DecisionTreeExperiment(name="Wages greater than 50k", target=data.target_classification_more_than_50_k,
                               max_tree_depth=12, prune_to=prune_to),
        DecisionTreeExperiment(name="Wages", target=data.target_classification,
                               max_tree_depth=10, prune_to=prune_to)
    ]


def generate_boosted_tree_experiments_for_car_data(data):
    return [
        BoostedTreeExperiment(name="Very Good Cars", target=data.target_classification_very_good,
                              max_tree_depth=10),
        BoostedTreeExperiment(name="Good Cars", target=data.target_classification_good, max_tree_depth=12),
        BoostedTreeExperiment(name="Acceptable Cars", target=data.target_classification_acceptable,
                              max_tree_depth=10),
        BoostedTreeExperiment(name="UnAcceptable Cars", target=data.target_classification_unacceptable,
                              max_tree_depth=10),
        BoostedTreeExperiment(name="Cars", target=data.target_classification, max_tree_depth=13)
    ]


def generate_boosted_tree_experiments_for_wage_data(data):
    return [
        BoostedTreeExperiment(name="Wages 50k or less", target=data.target_classification_50_k_or_less,
                              max_tree_depth=10),
        BoostedTreeExperiment(name="Wages greater than 50k", target=data.target_classification_more_than_50_k,
                              max_tree_depth=12),
        BoostedTreeExperiment(name="Wages", target=data.target_classification,
                              max_tree_depth=10)
    ]


def generate_knn_experiments_for_car_data(data):
    min_k = 1
    max_k = 10

    return [KnnExperiment(name="Very Good Cars", target=data.target_classification_very_good,
                          min_k=min_k, max_k=max_k),
            KnnExperiment(name="Good Cars", target=data.target_classification_good,
                          min_k=min_k, max_k=max_k),
            KnnExperiment(name="Acceptable Cars", target=data.target_classification_acceptable,
                          min_k=min_k, max_k=max_k),
            KnnExperiment(name="UnAcceptable Cars", target=data.target_classification_unacceptable,
                          min_k=min_k, max_k=max_k),
            KnnExperiment(name="Cars", target=data.target_classification,
                          min_k=min_k, max_k=max_k)
            ]


def generate_knn_experiments_for_wage_data(data):
    min_k = 1
    max_k = 50

    return [

            # KnnExperiment(name="Wages 50k or less", target=data.target_classification_50_k_or_less,
            #               min_k=min_k, max_k=max_k),
            # KnnExperiment(name="Wages greater than 50k", target=data.target_classification_more_than_50_k,
            #               min_k=min_k, max_k=max_k),
            KnnExperiment(name="Wages", target=data.target_classification,
                          min_k=min_k, max_k=max_k)
            ]


def generate_neural_network_experiments_for_car_data(data):
    min_number_of_layers = 3
    max_number_of_layers = 3

    return [
            # NeuralNetworkExperiment(name="Very Good Cars", target=data.target_classification_very_good,
            #                         min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            # NeuralNetworkExperiment(name="Good Cars", target=data.target_classification_good,
            #                         min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            # NeuralNetworkExperiment(name="Acceptable Cars", target=data.target_classification_acceptable,
            #                         min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            # NeuralNetworkExperiment(name="UnAcceptable Cars", target=data.target_classification_unacceptable,
            #                         min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            NeuralNetworkExperiment(name="of Cars", target=data.target_classification,
                                    min_layers=min_number_of_layers, max_layers=max_number_of_layers)]


def generate_neural_network_experiments_for_wage_data(data):
    min_number_of_layers = 2
    max_number_of_layers = 5

    return [NeuralNetworkExperiment(name="Wages 50k or less", target=data.target_classification_50_k_or_less,
                                    min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            NeuralNetworkExperiment(name="Wages greater than 50k", target=data.target_classification_more_than_50_k,
                                    min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            NeuralNetworkExperiment(name="Wages", target=data.target_classification,
                                    min_layers=min_number_of_layers, max_layers=max_number_of_layers),
            ]


def generate_svm_experiments_for_car_data(data):
    min_degree = 3
    max_degree = 4

    return [SVMExperiment(name="Very Good Cars", target=data.target_classification_very_good,
                          min_degree=min_degree, max_degree=max_degree),
            SVMExperiment(name="Good Cars", target=data.target_classification_good,
                          min_degree=min_degree, max_degree=max_degree),
            SVMExperiment(name="Acceptable Cars", target=data.target_classification_acceptable,
                          min_degree=min_degree, max_degree=max_degree),
            SVMExperiment(name="UnAcceptable Cars", target=data.target_classification_unacceptable,
                          min_degree=min_degree, max_degree=max_degree),
            SVMExperiment(name="Cars", target=data.target_classification,
                          min_degree=min_degree, max_degree=max_degree)
            ]


def generate_svm_experiments_for_wage_data(data):
    min_degree = 3
    max_degree = 3

    return [

            SVMExperiment(name="Wages", target=data.target_classification,
                          min_degree=min_degree, max_degree=max_degree)
            ]


def run_decision_tree_experiments(data_folder, features, experiments):
    experiment_title = "Decision Tree classification of"

    learner = DecisionTreeLeaner('./report_artifacts/' + data_folder + '/figures/decision_tree/',
                                 './report_artifacts/' + data_folder + '/data/decision_tree/')

    for e in experiments:
        learner.run_decision_tree_classification_experiment("{} {}".format(experiment_title, e.name),
                                                            features, e.target,
                                                            prune_factor=e.prune_to, max_tree_depth=e.max_tree_depth)


def run_boosted_tree_experiments(data_folder, features, experiments):
    experiment_title = "Boosted Tree classification of"

    learner = DecisionTreeLeaner('./report_artifacts/' + data_folder + '/figures/boosted_tree/',
                                 './report_artifacts/' + data_folder + '/data/boosted_tree/')

    for e in experiments:
        learner.run_boosted_tree_classification_experiment("{} {}".format(experiment_title, e.name),
                                                           features, e.target,
                                                           max_tree_depth=e.max_tree_depth)


def run_knn_experiments(data_folder, features, experiments):
    sns.set(font_scale=.60)

    experiment_title = "K-NN classification of"

    learner = KnnLearner('./report_artifacts/' + data_folder + '/figures/knn/',
                         './report_artifacts/' + data_folder + '/data/knn/')

    for e in experiments:
        learner.run_knn_experiment("{} {}".format(experiment_title, e.name), features, e.target, e.min_k, e.max_k)

    sns.set(font_scale=1.0)

def run_neural_network_experiments(data_folder, features, experiments):
    sns.set(font_scale=.75)

    experiment_title = "Neural Network Classification"
    learner = NeuralNetworkLearner('./report_artifacts/' + data_folder + '/figures/neural_network/',
                                   './report_artifacts/' + data_folder + '/data/neural_network/')
    for e in experiments:
        learner.run_neural_network_experiment("{} {}".format(experiment_title, e.name),
                                              features, e.target, e.min_layers, e.max_layers)
    sns.set(font_scale=1.0)


def run_svm_experiments(data_folder, features, experiments):
    sns.set(font_scale=.75)

    experiment_title = "Support Vector Matrix Classification"
    learner = SVMLeaner('./report_artifacts/' + data_folder + '/figures/support_vector_machine/',
                        './report_artifacts/' + data_folder + '/data/support_vector_machine/')

    for e in experiments:
        learner.run_SVM_experiment("{} {}".format(experiment_title, e.name),
                                   features, e.target, e.min_degree, e.max_degree)

    sns.set(font_scale=1.0)


sns.set(style="whitegrid")

car_data = CarData()
wage_data = AdultData()

car_data_folder = 'car_data'
wage_data_folder = 'wage_data'

car_data.generate_csv()

#run_decision_tree_experiments(car_data_folder, car_data.features, generate_decision_tree_experiments_for_car_data(car_data))
#run_boosted_tree_experiments(car_data_folder, car_data.features, generate_boosted_tree_experiments_for_car_data(car_data))

# run_decision_tree_experiments(wage_data_folder, wage_data.features, generate_decision_tree_experiments_for_wage_data(wage_data))
#run_boosted_tree_experiments(wage_data_folder, wage_data.features,
  #                            generate_boosted_tree_experiments_for_wage_data(wage_data))

# run_knn_experiments(car_data_folder, car_data.features, generate_knn_experiments_for_car_data(car_data))
#run_knn_experiments(wage_data_folder, wage_data.features, generate_knn_experiments_for_wage_data(wage_data))

run_neural_network_experiments(car_data_folder, car_data.features,
                                generate_neural_network_experiments_for_car_data(car_data))
# run_neural_network_experiments(wage_data_folder, wage_data.features,
#                                generate_neural_network_experiments_for_wage_data(wage_data))

#run_svm_experiments(car_data_folder, car_data.features, generate_svm_experiments_for_car_data(car_data))
#run_svm_experiments(wage_data_folder, wage_data.features, generate_svm_experiments_for_wage_data(wage_data))

#car_data.generate_target_distribution_plot()
#wage_data.generate_target_distribution_plot()
