# ALL IMPORTS HERE
from src.metrics import precision_at_k
from src.recommenders import MainRecommender
import pandas as pd
import numpy as np
import yaml
import sys
import inspect
from pprint import pprint
import warnings

warnings.filterwarnings('ignore')


def load_settings(need_print=False):
    """
    - get data from settings.yaml as variables;
    - print all imported data (class, function, constants)
    :return constants as dictionary
    """
    global _CONSTANTS
    _CONSTANTS = yaml.load(open("settings.yaml", 'r'), Loader=yaml.FullLoader)
    if need_print:
        print('Loaded following classes:')
        pprint([obj[0] for obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)])
        print('\nLoaded following functions:')
        pprint([obj[0] for obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)])
        print(f"\nLoaded following constants:")
        [print(f'{key:20} = {val:20} | {type(val)}') for key, val in _CONSTANTS.items()]
    return _CONSTANTS


class Dataset:
    data_train = pd.read_csv('../_ADDS/webinar_6/data_init/retail_train.csv')
    data_test = pd.read_csv(r'../_ADDS/webinar_8/retail_test1.csv')
    item_features = pd.read_csv('../_ADDS/webinar_6/data_init/product.csv')
    user_features = pd.read_csv('../_ADDS/webinar_6/data_init/hh_demographic.csv')
    _CONSTANTS = load_settings()

    def data_prefilter(self, make_worse=False):
        # column start processing
        self.item_features.columns = [col.lower() for col in self.item_features.columns]
        self.user_features.columns = [col.lower() for col in self.user_features.columns]
        self.item_features.rename(columns={'product_id': _CONSTANTS['ITEM_COL']}, inplace=True)
        self.user_features.rename(columns={'household_key': _CONSTANTS['USER_COL']}, inplace=True)

        # mark unpopular departments
        department_size = pd.DataFrame(self.item_features. \
                                       groupby('department')[_CONSTANTS['ITEM_COL']].nunique(). \
                                       sort_values(ascending=False)).reset_index()
        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 10].department.tolist()
        items_in_rare_departments = self.item_features[self.item_features['department']. \
            isin(rare_departments)].item_id.unique().tolist()

        # transform categorical data to numeric
        self.item_features['brand'] = pd.factorize(self.item_features['brand'])[0]
        self.item_features['commodity_type'] = pd.factorize(self.item_features['commodity_desc'])[0]

        # recalculate mean age, income of users
        self.user_features.loc[self.user_features['age_desc'] == '65+', 'age'] = 75
        self.user_features.loc[self.user_features['age_desc'] != '65+', 'age'] = self.user_features.loc[
            self.user_features['age_desc'] != '65+', 'age_desc'].apply(
            lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1])) / 2))

        self.user_features.loc[self.user_features['income_desc'].str.contains('\+'), 'income'] = 300
        self.user_features.loc[self.user_features['income_desc'].str.contains('Under'), 'income'] = 10
        self.user_features.loc[self.user_features['income_desc'].str.contains('-'), 'income'] = self.user_features.loc[
            self.user_features['income_desc'].str.contains('-'), 'income_desc'].apply(
            lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1][:-1])) / 2))
        # calculating adults_num and has_kids
        self.user_features.loc[(self.user_features['hh_comp_desc'].str.contains('Kids')) & \
                               (~self.user_features['hh_comp_desc'].str.contains('No')), 'has_kids'] = 1
        self.user_features.loc[self.user_features['has_kids'].isnull(), 'has_kids'] = 0
        self.user_features.loc[self.user_features['hh_comp_desc'].str.contains('Adults'), 'adults_num'] = 2
        self.user_features.loc[self.user_features['hh_comp_desc'].str.contains('Single'), 'adults_num'] = 1
        self.user_features.loc[self.user_features['hh_comp_desc'].str.contains('1 Adult'), 'adults_num'] = 1
        self.user_features.loc[self.user_features['hh_comp_desc'] == 'Unknown', 'adults_num'] = 1

        # remove '+' from category and make col type numeric
        self.user_features.loc[self.user_features['household_size_desc'] == '5+', 'household_size_desc'] = 5
        self.user_features.loc[self.user_features['kid_category_desc'] == '3+', 'kid_category_desc'] = 3
        self.user_features.loc[self.user_features['kid_category_desc'] == 'None/Unknown', 'kid_category_desc'] = 0
        self.user_features['household_size_desc'] = self.user_features['household_size_desc'].astype(int)
        self.user_features['kid_category_desc'] = self.user_features['kid_category_desc'].astype(int)

        # transform categorical data to numeric
        self.user_features = pd.concat([self.user_features, pd.get_dummies(self.user_features['homeowner_desc'])],
                                       axis=1)
        self.user_features = pd.concat([self.user_features, pd.get_dummies(self.user_features['marital_status_code'])],
                                       axis=1)

        # remove text data
        self.user_features = self.user_features.iloc[:, 5:]
        self.item_features = self.item_features[[self._CONSTANTS['ITEM_COL'],
                                                 'manufacturer',
                                                 'brand',
                                                 'commodity_type']]

        # iterate throw train and test
        for i in ['train', 'test']:
            if i == 'train':
                df = self.data_train.copy()
            else:
                df = self.data_test.copy()
            # remove unpopular departments
            df = df[~df[_CONSTANTS['ITEM_COL']].isin(items_in_rare_departments)]

            if make_worse:
                # making worst max precision score (hypothesis)
                # remove cheap ones
                df['price'] = df['sales_value'] / (np.maximum(df['quantity'], 1))
                df = df[df['price'] > 2]
                # remove expensive ones
                df = df[df['price'] < 50]

            # add new features
            # user's mean check
            basket_stat = self.user_features.merge(df, on=_CONSTANTS['USER_COL'], how='left')
            basket_stat = basket_stat.pivot_table(index=_CONSTANTS['USER_COL'], values=['basket_id', 'sales_value'],
                                                  aggfunc={'basket_id': 'count', 'sales_value': 'sum'})
            basket_stat = basket_stat['sales_value'] / basket_stat['basket_id']
            basket_stat = basket_stat.reset_index()
            basket_stat.rename(columns={0: 'avg_check'}, inplace=True)
            df = df.merge(basket_stat.reset_index(), on=_CONSTANTS['USER_COL'])
            del basket_stat

            # get top popularity items
            df = df.merge(df.groupby(_CONSTANTS['ITEM_COL'])['quantity'].sum().reset_index(),
                          on=_CONSTANTS['ITEM_COL'],
                          how='left',
                          suffixes=['', '_total'])

            # remove super unpopular items over 12 month
            max_day = df['day'].max()
            items_365 = df.loc[
                (df['day'] <= max_day) & (df['day'] >= max_day - 365), _CONSTANTS['ITEM_COL']].unique().tolist()
            df = df.loc[df[_CONSTANTS['ITEM_COL']].isin(items_365)]
            del items_365

            # change item_id to fakes where we think user "already" served his needs
            df.loc[df['quantity_total'] >= _CONSTANTS['TAKE_N_POPULAR'], _CONSTANTS['ITEM_COL']] = 999999

            # commit instance changes
            if i == 'train':
                self.data_train = df.copy()
            else:
                self.data_test = df.copy()
            del df

    def data_split(self,
                   val_lvl_1_size_weeks=_CONSTANTS['VAL_MATCHER_WEEKS'],
                   val_lvl_2_size_weeks=_CONSTANTS['VAL_RANKER_WEEKS']):

        self.data_train_lvl_1 = self.data_train[self.data_train['week_no'] < self.data_train['week_no'].max() - \
                                                (val_lvl_1_size_weeks + val_lvl_2_size_weeks)]
        self.data_val_lvl_1 = self.data_train[(self.data_train['week_no'] >= self.data_train['week_no'].max() - \
                                               (val_lvl_1_size_weeks + val_lvl_2_size_weeks)) & \
                                              (self.data_train['week_no'] < self.data_train['week_no'].max() - \
                                               (val_lvl_2_size_weeks))]

        self.data_train_lvl_2 = self.data_val_lvl_1.copy()
        self.data_val_lvl_2 = self.data_train[self.data_train['week_no'] >= self.data_train['week_no'].max() - \
                                              val_lvl_2_size_weeks]

        self.result_lvl_1 = self.data_val_lvl_1.groupby(_CONSTANTS['USER_COL'])[
            _CONSTANTS['ITEM_COL']].unique().reset_index()
        self.result_lvl_1.columns = [_CONSTANTS['USER_COL'], _CONSTANTS['ACTUAL_COL']]

        self.users_train = self.data_train_lvl_1[_CONSTANTS['USER_COL']].tolist()
        self.users_valid = self.result_lvl_1[_CONSTANTS['USER_COL']].tolist()
        self.new_users = list(set(self.users_valid) - set(self.users_train))
        self.all_users = list(set(self.users_valid) & set(self.users_train))
        self.result_lvl_1 = self.result_lvl_1[~self.result_lvl_1[_CONSTANTS['USER_COL']].isin(self.new_users)]

    def postfilter_items(user_id, recommendations):
        pass
