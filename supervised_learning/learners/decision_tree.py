import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class DecisionTreeLeaner:

    def __init__(self, figure_path, csv_path):
        self.figure_path = figure_path
        self.csv_path = csv_path

    def prune(self, d_tree, min_samples_leaf=1):
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

    def get_train_test_split(self, df_features, df_target):
        return train_test_split(df_features, df_target, test_size=0.33, random_state=42)

    def get_tree_details(self, name, tree_score, d_tree, fit_time, score_time, print_details=True):
        if print_details:
            print 'Results for {}\n Accuracy: {}\n Tree Splitting: {}\n Tree Depth: {}\n Tree Nodes: {}\n Fit Time: {}\n Score Time: {}\n' \
                .format(name, tree_score, d_tree.splitter, d_tree.tree_.max_depth, d_tree.tree_.node_count, fit_time,
                        score_time)

        return pd.Series(dict(name=name, score=tree_score, splitter=d_tree.splitter, depth=d_tree.tree_.max_depth,
                              nodes=d_tree.tree_.node_count, fit_time=fit_time, score_time=score_time))

    def get_boosted_tree_details(self, name, tree_score, d_tree, fit_time, score_time, print_details=True):
        if print_details:
            print 'Results for {}\n Accuracy: {}\n Tree Depth: {}\n Fit Time: {}\n Score Time: {}\n' \
                .format(name, tree_score, d_tree.max_depth, fit_time, score_time)

        return pd.Series(dict(name=name, score=tree_score, depth=d_tree.max_depth,
                              fit_time=fit_time, score_time=score_time))

    def decision_tree_experiment(self, ex_data, splitter, max_depth, print_details=True, prune_factor=0):
        x_train, y_train, x_test, y_test = ex_data['x_train'], ex_data['y_train'], ex_data['x_test'], ex_data['y_test']
        d_tree = tree.DecisionTreeClassifier(splitter=splitter, max_depth=max_depth)

        start = time.time()
        d_tree.fit(x_train, y_train)
        end = time.time()
        fit_time = (end - start)

        if prune_factor > 1:
            self.prune(d_tree, min_samples_leaf=prune_factor)

        start = time.time()
        score = d_tree.score(x_test, y_test)
        end = time.time()
        score_time = (end - start)

        details = self.get_tree_details("{} Tree w/ depth of {}".format(splitter, max_depth),
                                        score, d_tree, fit_time=fit_time, score_time=score_time,
                                        print_details=print_details)

        return details

    def xgboost_tree_experiment(self, ex_data, max_depth, print_details=True):
        boost_tree = XGBClassifier(max_depth=max_depth)

        x_train, y_train, x_test, y_test = ex_data['x_train'], ex_data['y_train'], ex_data['x_test'], ex_data['y_test']

        start = time.time()
        boost_tree.fit(x_train, y_train)
        end = time.time()
        fit_time = (end - start)

        start = time.time()
        score = boost_tree.score(x_test, y_test)
        end = time.time()
        score_time = (end - start)

        details = self.get_boosted_tree_details("Tree w/ depth of {}".format(max_depth),
                                                score, boost_tree, fit_time=fit_time, score_time=score_time,
                                                print_details=print_details)

        return details

    def run_decision_tree_classification_experiment(self, name, df_features, series_target, min_tree_depth=1,
                                                    max_tree_depth=10,
                                                    prune_factor=0):
        x_train, x_test, y_train, y_test = self.get_train_test_split(df_features, series_target)
        data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        depth_range = range(min_tree_depth, max_tree_depth + 1)

        best_experiment = []
        random_experiment = []

        for depth in depth_range:
            best_experiment.append(self.decision_tree_experiment(data,
                                                                 splitter='best', max_depth=depth,
                                                                 prune_factor=prune_factor))

            random_experiment.append(self.decision_tree_experiment(data,
                                                                   splitter='random', max_depth=depth,
                                                                   prune_factor=prune_factor))

        df_best = pd.DataFrame(best_experiment)
        df_random = pd.DataFrame(random_experiment)

        df_graph = pd.concat([df_best, df_random])
        df_graph.to_csv(self.csv_path + name.lower().replace(' ', '_') + '.csv', index=False,
                        index_label='depth')

        ax = sns.barplot(x="depth", y="score", hue="splitter", data=df_graph)
        ax.set_title(name)
        ax.set_ylabel("accuracy")
        ax.set_xlabel("tree depth")

        plt.ylim(df_graph.score.min() - (df_graph.score.min() * .01), df_graph.score.max())

        fig = ax.get_figure()
        fig.savefig(self.figure_path + name.lower().replace(' ', '_') + '.png')

        plt.show()

    def run_boosted_tree_classification_experiment(self, name, df_features, series_target, min_tree_depth=1,
                                                   max_tree_depth=10):
        x_train, x_test, y_train, y_test = self.get_train_test_split(df_features, series_target)
        data = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        depth_range = range(min_tree_depth, max_tree_depth + 1)

        boost_experiment = []

        for depth in depth_range:
            boost_experiment.append(self.xgboost_tree_experiment(data, max_depth=depth))

        df_boost = pd.DataFrame(boost_experiment)

        df_boost.to_csv(self.csv_path + name.lower().replace(' ', '_') + '.csv',
                        index=False, index_label='depth')

        ax = sns.barplot(x="depth", y="score", data=df_boost)
        ax.set_title(name)
        ax.set_ylabel("accuracy")
        ax.set_xlabel("tree depth")

        plt.ylim(df_boost.score.min() - (df_boost.score.min() * .01), df_boost.score.max())

        fig = ax.get_figure()
        fig.savefig(self.figure_path + name.lower().replace(' ', '_') + '.png')

        plt.show()
