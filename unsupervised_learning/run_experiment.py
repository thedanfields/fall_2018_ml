import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, validation_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from car_data.cars_data import CarData
from wage_data.adult_data import WageData


def plot_learning_curve(name, data_type, train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('accuracy')
    plt.grid(ls='--')
    plt.legend(loc='best')

    plt.savefig("./plots/learning/" + data_type + "/" + name.lower().replace(' ', '_') + '.png', bbox_inches='tight')

    plt.show()


def plot_validation_curve(name, data_type, param_range, train_scores, test_scores, title, alpha=0.1):

    sort_idx = np.argsort(param_range)
    param_range=np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    plt.savefig("./plots/validation/" + data_type + "/" + name.lower().replace(' ', '_') + '.png', bbox_inches='tight')

    plt.show()


def run_neural_learner(data_type, df_features, series_target, title):

    X_train, X_test, y_train, y_test = train_test_split(df_features, series_target, random_state=0)

    classifier = MLPClassifier()

    pipeline = Pipeline(steps=[('clf', classifier)])

    cv = StratifiedKFold(n_splits=5, random_state=42)

    param_grid = {'clf__hidden_layer_sizes': [(15), (15, 15), (15, 15, 15), (15, 15, 15, 15)],
                  'clf__max_iter': [1000, 2000, 3000, 4000, 5000]
                  }

    rg_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
    rg_cv.fit(X_train, y_train)
    print(title)
    print("Tuned rg best params: {}".format(rg_cv.best_params_))

    ypred = rg_cv.predict(X_train)
    print(classification_report(y_train, ypred))
    print('######################')
    ypred2 = rg_cv.predict(X_test)
    print(classification_report(y_test, ypred2))

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=rg_cv.best_estimator_, X=X_train, y=y_train, cv=cv, scoring='accuracy')

    plot_learning_curve(title, data_type, train_sizes, train_scores, test_scores,
                        title='Learning curve for Neural Network Classification w/ ' + title)

    param_range1 = [(15), (15,15), (15,15,15), (15,15, 15, 15)]

    train_scores, test_scores = validation_curve(
        estimator=rg_cv.best_estimator_, X=X_train, y=y_train, param_name="clf__hidden_layer_sizes",
        param_range=param_range1,
        cv=cv, scoring="accuracy")

    plot_validation_curve(title, data_type, [1,2,3,4], train_scores, test_scores,
                          title="Validation Curve for Neural Network Hidden Layers w/ " + title, alpha=0.1)

    print("######################")
    print("")


def run_car_experiments():

    car_data = CarData()

    #run_best_car_learner(car_data.features, car_data.target_classification, "No reduction")

    component_range = range(1, 10)
    data_type = "car"

    for n in component_range:
        print '! ' + str(n)
        pci_reduced_data, _ = car_data.reduce_via_pca(n)
        ica_reduced_data, _ = car_data.reduce_via_ica(n)
        rando_reduced_data, _ = car_data.reduce_via_random_projection(n)
        feature_agglomerated_data, _ = car_data.reduce_via_feature_agglomeration(n)

        run_neural_learner(data_type, pci_reduced_data, car_data.target_classification, "PCI reduction to " + str(n))
        run_neural_learner(data_type, ica_reduced_data, car_data.target_classification, "ICA reduction to " + str(n))
        run_neural_learner(data_type, rando_reduced_data, car_data.target_classification, "Random Projection reduction to " + str(n))
        run_neural_learner(data_type, feature_agglomerated_data, car_data.target_classification, "Feature Agglomerated reduction to " + str(n))

        print '! ' + str(n) + ' !'


def run_wage_experiments():
    wage_data = WageData()

    component_range = range(1, 50)

    data_type = "wage"

    for n in component_range:
        print '! ' + str(n)
        pci_reduced_data, _ = wage_data.reduce_via_pca(n)
        ica_reduced_data, _ = wage_data.reduce_via_ica(n)
        rando_reduced_data, _ = wage_data.reduce_via_random_projection(n)
        feature_agglomerated_data, _ = wage_data.reduce_via_feature_agglomeration(n)

        run_neural_learner(data_type, pci_reduced_data, wage_data.target_classification, "PCI reduction to " + str(n))
        run_neural_learner(data_type, ica_reduced_data, wage_data.target_classification, "ICA reduction to " + str(n))
        run_neural_learner(data_type, rando_reduced_data, wage_data.target_classification,
                           "Random Projection reduction to " + str(n))
        run_neural_learner(data_type, feature_agglomerated_data, wage_data.target_classification,
                           "Feature Agglomerated reduction to " + str(n))

        print '! ' + str(n) + ' !'


run_car_experiments()
run_wage_experiments()