import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def prune(d_tree, min_samples_leaf=1):
    if d_tree.min_samples_leaf >= min_samples_leaf:
        raise Exception('Tree already more pruned')
    else:
        d_tree.min_samples_leaf = min_samples_leaf
        tree = d_tree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i] = -1
                tree.children_right[i] = -1


def get_train_test_split(df_features, df_target):
    return train_test_split(df_features, df_target, test_size=0.33, random_state=42)


def get_tree_details(name, tree_score, d_tree, fit_time, score_time, print_details=True):
    if print_details:
        print 'Results for {}\n Accuracy: {}\n Tree Splitting: {}\n Tree Depth: {}\n Tree Nodes: {}\n Fit Time: {}\n Score Time: {}\n' \
            .format(name, tree_score, d_tree.splitter, d_tree.tree_.max_depth, d_tree.tree_.node_count, fit_time,
                    score_time)

    return pd.Series(dict(name=name, score=tree_score, splitter=d_tree.splitter, depth=d_tree.tree_.max_depth,
                          nodes=d_tree.tree_.node_count, fit_time=fit_time, score_time=score_time))


def decision_tree_experiment(ex_data, splitter, max_depth, print_details=True, prune_factor=0):
    d_tree = tree.DecisionTreeClassifier(splitter=splitter, max_depth=max_depth)

    start = time.time()
    d_tree.fit(ex_data['x_train'], ex_data['y_train'])
    end = time.time()
    fit_time = (end - start)

    if prune_factor > 1:
        prune(d_tree, min_samples_leaf=prune_factor)

    start = time.time()
    score = d_tree.score(ex_data['x_test'], ex_data['y_test'])
    end = time.time()
    score_time = (end - start)

    details = get_tree_details("{} Tree w/ depth of {}".format(splitter, max_depth),
                               score, d_tree, fit_time=fit_time, score_time=score_time, print_details=print_details)

    return details


def run_classification_experiment(name, df_features, series_target, min_tree_depth=1, max_tree_depth=10,
                                  prune_factor=0):
    x_train, x_test, y_train, y_test = get_train_test_split(df_features, series_target)
    data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    depth_range = range(min_tree_depth, max_tree_depth + 1)

    best_experiment = []
    random_experiment = []

    for depth in depth_range:
        best_experiment.append(decision_tree_experiment(data,
                                                        splitter='best', max_depth=depth, prune_factor=prune_factor))

        random_experiment.append(decision_tree_experiment(data,
                                                          splitter='random', max_depth=depth,
                                                          prune_factor=prune_factor))

    df_best = pd.DataFrame(best_experiment)
    df_random = pd.DataFrame(random_experiment)

    df_graph = pd.concat([df_best, df_random])
    df_graph.to_csv('./report_artifacts/data/' + name.lower().replace(' ', '_') + '.csv', index=False, index_label='depth')

    ax = sns.barplot(x="depth", y="score", hue="splitter", data=df_graph)
    ax.set_title(name)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("tree depth")

    plt.ylim(df_graph.score.min() - (df_graph.score.min() * .01), df_graph.score.max())

    fig = ax.get_figure()
    fig.savefig('./report_artifacts/figures/' + name.lower().replace(' ', '_') + '.png')

    plt.show()
    pass


def genereate_target_distribution_plot(dataframe):
    ax = sns.countplot(x='classification',
                       data=dataframe,
                       order=dataframe['classification'].value_counts().index)
    ax.set_title('Distribution of Cars by Classification')
    fig = ax.get_figure()
    fig.savefig('./report_artifacts/figures/classification_distro.png')
    plt.show()


input_file = "car.data.csv"
df = pd.read_csv(input_file, header=0)
df_dummies_features = pd.get_dummies(df,
                                     columns=['buying_price', 'maintenance_cost', 'number_of_doors',
                                              'carrying_capacity',
                                              'trunk_size', 'safety_rating'],
                                     drop_first=True)

df_dummies_classification = pd.get_dummies(pd.DataFrame(df.classification), columns=['classification'])

all_feature_columns = ['buying_price_low',
                       'buying_price_med',
                       'buying_price_vhigh',
                       'maintenance_cost_low',
                       'maintenance_cost_med',
                       'maintenance_cost_vhigh',
                       'number_of_doors_3',
                       'number_of_doors_4',
                       'number_of_doors_5more',
                       'carrying_capacity_4',
                       'carrying_capacity_more',
                       'trunk_size_med',
                       'trunk_size_small',
                       'safety_rating_low',
                       'safety_rating_med']

df_features = pd.DataFrame(df_dummies_features, columns=all_feature_columns)

genereate_target_distribution_plot(df)

prune_to = 10
max_tree_depth = 15

run_classification_experiment("Classification of Very Good Cars",
                              df_features, df_dummies_classification.classification_vgood,
                              prune_factor=prune_to, max_tree_depth=10)
run_classification_experiment("Classification of Good Cars",
                              df_features, df_dummies_classification.classification_good,
                              prune_factor=prune_to, max_tree_depth=12)
run_classification_experiment("Classification of Acceptable Cars",
                              df_features, df_dummies_classification.classification_acc,
                              prune_factor=prune_to, max_tree_depth=10)
run_classification_experiment("Classification of Unacceptable Cars",
                              df_features, df_dummies_classification.classification_unacc,
                              prune_factor=prune_to, max_tree_depth=10)

run_classification_experiment("Classification of Cars",
                              df_features, df.classification.astype("category").cat.codes,
                              prune_factor=prune_to, max_tree_depth=13)
