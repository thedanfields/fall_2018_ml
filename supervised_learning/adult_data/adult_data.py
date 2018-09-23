import pandas as pd
import seaborn as sns
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split


class AdultData:
    df_raw = None  # type: Union[Union[TextFileReader, DataFrame], Any]
    target_classification = None  # type: object
    features = None  # type: DataFrame

    def __init__(self):
        input_file = "./adult_data/adult.data.csv"
        self.df_raw = pd.read_csv(input_file, header=0)
        df_dummies_features = pd.get_dummies(self.df_raw,
                                             columns=['workclass', 'education', 'maritalstatus', 'occupation',
                                                      'relationship', 'race', 'sex', 'nativecountry'],
                                             drop_first=True)

        df_dummies_classification = pd.get_dummies(pd.DataFrame(self.df_raw.classification), columns=['classification'])

        self.target_classification_50_k_or_less = df_dummies_classification['classification_50_k_or_less']
        self.target_classification_more_than_50_k = df_dummies_classification['classification_greater_than_50_k']

        self.target_classification = self.df_raw.classification.astype("category").cat.codes

        all_feature_columns = ['age', 'cpsfinalweight', 'educationnum', 'capitalgain', 'capitalloss', 'hours_per_week',
                               'workclass_Federal_gov', 'workclass_Local_gov',
                               'workclass_Never_worked', 'workclass_Private', 'workclass_Self_emp_inc',
                               'workclass_Self_emp_not_inc', 'workclass_State_gov', 'workclass_Without_pay',
                               'education_11th', 'education_12th', 'education_1st_4th', 'education_5th_6th',
                               'education_7th_8th', 'education_9th', 'education_Assoc_acdm', 'education_Assoc_voc',
                               'education_Bachelors', 'education_Doctorate', 'education_HS_grad', 'education_Masters',
                               'education_Preschool', 'education_Prof_school', 'education_Some_college',
                               'maritalstatus_Married_AF_spouse', 'maritalstatus_Married_civ_spouse',
                               'maritalstatus_Married_spouse_absent', 'maritalstatus_Never_married',
                               'maritalstatus_Separated', 'maritalstatus_Widowed', 'occupation_Adm_clerical',
                               'occupation_Armed_Forces', 'occupation_Craft_repair', 'occupation_Exec_managerial',
                               'occupation_Farming_fishing', 'occupation_Handlers_cleaners',
                               'occupation_Machine_op_inspct', 'occupation_Other_service', 'occupation_Priv_house_serv',
                               'occupation_Prof_specialty', 'occupation_Protective_serv', 'occupation_Sales',
                               'occupation_Tech_support', 'occupation_Transport_moving', 'relationship_Not_in_family',
                               'relationship_Other_relative', 'relationship_Own_child', 'relationship_Unmarried',
                               'relationship_Wife', 'race_Asian_Pac_Islander', 'race_Black', 'race_Other', 'race_White',
                               'sex_Male', 'nativecountry_Cambodia', 'nativecountry_Canada', 'nativecountry_China',
                               'nativecountry_Columbia', 'nativecountry_Cuba', 'nativecountry_Dominican_Republic',
                               'nativecountry_Ecuador', 'nativecountry_El_Salvador', 'nativecountry_England',
                               'nativecountry_France', 'nativecountry_Germany', 'nativecountry_Greece',
                               'nativecountry_Guatemala', 'nativecountry_Haiti', 'nativecountry_Holand_Netherlands',
                               'nativecountry_Honduras', 'nativecountry_Hong', 'nativecountry_Hungary',
                               'nativecountry_India', 'nativecountry_Iran', 'nativecountry_Ireland',
                               'nativecountry_Italy', 'nativecountry_Jamaica', 'nativecountry_Japan',
                               'nativecountry_Laos', 'nativecountry_Mexico', 'nativecountry_Nicaragua',
                               'nativecountry_Outlying_US(Guam_USVI_etc)', 'nativecountry_Peru',
                               'nativecountry_Philippines', 'nativecountry_Poland', 'nativecountry_Portugal',
                               'nativecountry_Puerto_Rico', 'nativecountry_Scotland', 'nativecountry_South',
                               'nativecountry_Taiwan', 'nativecountry_Thailand', 'nativecountry_Trinadad&Tobago',
                               'nativecountry_United_States', 'nativecountry_Vietnam', 'nativecountry_Yugoslavia']

        self.features = pd.DataFrame(df_dummies_features, columns=all_feature_columns)

    def generate_target_distribution_plot(self):
        ax = sns.countplot(x='classification',
                           data=self.df_raw,
                           order=self.df_raw['classification'].value_counts().index)
        ax.set_title('Distribution of Wages by Classification')
        fig = ax.get_figure()
        fig.savefig('./report_artifacts/wage_data/figures/classification_distro.png')
        # plt.show()

    @staticmethod
    def get_train_test_split(df_features, df_target):
        return train_test_split(df_features, df_target, test_size=0.33, random_state=42)
