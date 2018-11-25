import pandas as pd
import seaborn as sns
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

import time

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration

class CarData:
    df_raw = None  # type: Union[Union[TextFileReader, DataFrame], Any]
    target_classification = None  # type: object
    features = None  # type: DataFrame

    def __init__(self):
        input_file = "./car_data/car.data.csv"
        self.df_raw = pd.read_csv(input_file, header=0)
        df_dummies_features = pd.get_dummies(self.df_raw,
                                             columns=['buying_price', 'maintenance_cost', 'number_of_doors',
                                                      'carrying_capacity',
                                                      'trunk_size', 'safety_rating'],
                                             drop_first=True)

        df_dummies_classification = pd.get_dummies(pd.DataFrame(self.df_raw.classification), columns=['classification'])

        self.target_classification_unacceptable = df_dummies_classification.classification_unacc
        self.target_classification_acceptable = df_dummies_classification.classification_acc
        self.target_classification_good = df_dummies_classification.classification_good
        self.target_classification_very_good = df_dummies_classification.classification_vgood
        self.target_classification = self.df_raw.classification.astype("category").cat.codes

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

        self.features = pd.DataFrame(df_dummies_features, columns=all_feature_columns)

        data = pd.concat([self.target_classification, self.features], axis=1)
        data.rename(columns={0: 'target_category'}, inplace=True)

        self.data = data


    def generate_target_distribution_plot(self):
        ax = sns.countplot(x='classification',
                           data=self.df_raw,
                           order=self.df_raw['classification'].value_counts().index)
        ax.set_title('Distribution of Cars by Classification')
        fig = ax.get_figure()
        fig.savefig('./report_artifacts/car_data/figures/classification_distro.png')
        #plt.show()

    def generate_csv(self):
        encoded_data_all = self.features.assign(classification=self.target_classification.values)
        encoded_data_all.to_csv('./report_artifacts/car_data/encoded_data_all.csv', index=False)

        encoded_data_unacceptable = self.features.assign(classification=self.target_classification_unacceptable.values)
        encoded_data_unacceptable.to_csv('./report_artifacts/car_data/encoded_data_unacceptable.csv', index=False)

        encoded_data_acceptable = self.features.assign(classification=self.target_classification_acceptable.values)
        encoded_data_acceptable.to_csv('./report_artifacts/car_data/encoded_data_acceptable.csv', index=False)

        encoded_data_good  = self.features.assign(classification=self.target_classification_good.values)
        encoded_data_good.to_csv('./report_artifacts/car_data/encoded_data_good.csv', index=False)

        encoded_data_vgood = self.features.assign(classification=self.target_classification_very_good.values)
        encoded_data_vgood.to_csv('./report_artifacts/car_data/encoded_data_very_good.csv', index=False)

    def reduce_via_pca(self, number_of_components):
        pca = PCA(n_components=number_of_components)

        start = time.time()
        principal_components = pca.fit_transform(self.features)
        end = time.time()
        fit_time = (end - start)

        return pd.DataFrame(data=principal_components), fit_time

    def reduce_via_ica(self, number_of_components):
        fast_ica = FastICA(n_components=number_of_components)

        start = time.time()
        principal_components = fast_ica.fit_transform(self.features)
        end = time.time()
        fit_time = (end - start)

        return pd.DataFrame(data=principal_components), fit_time

    def reduce_via_random_projection(self, number_of_components):
        rando_projection = GaussianRandomProjection(n_components=number_of_components)

        start = time.time()
        principal_components = rando_projection.fit_transform(self.features)
        end = time.time()
        fit_time = (end - start)

        return pd.DataFrame(data=principal_components), fit_time

    def reduce_via_feature_agglomeration(self, number_of_clusters):
        agglo = FeatureAgglomeration(n_clusters=number_of_clusters)

        start = time.time()
        agglomerated_data = agglo.fit_transform(self.features)
        end = time.time()
        fit_time = (end - start)

        return pd.DataFrame(data=agglomerated_data), fit_time
