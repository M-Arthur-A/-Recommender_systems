import catboost
from src.metrics import precision_at_k
import pandas as pd
import numpy as np
import yaml

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight
from catboost import Pool, CatBoostRanker


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data: pd.DataFrame, weighting: bool = True):
        self._CONSTANTS = yaml.load(open("settings.yaml", 'r'), Loader=yaml.FullLoader)
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby([self._CONSTANTS['USER_COL'], self._CONSTANTS['ITEM_COL']])[
            'quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases[self._CONSTANTS['ITEM_COL']] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby(self._CONSTANTS['ITEM_COL'])['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[
            self.overall_top_purchases[self._CONSTANTS['ITEM_COL']] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = \
            self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    def _prepare_matrix(self, data: pd.DataFrame):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index=self._CONSTANTS['USER_COL'],
                                          columns=self._CONSTANTS['ITEM_COL'],
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        own_recommender = ItemItemRecommender(K=1, num_threads=6)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        return own_recommender

    def preprocessing(self, dataset, t='train'):
        """
        Prepare data for ranker -- splitting in correct way
        :param dataset: object of Dataset class
        :param t: 'train' (data_train_lvl_2) or 'valuate' (data_val_lvl_2)
        :return: X and y
        """
        if t == 'train':
            df = dataset.data_train_lvl_2
        else:
            df = dataset.data_val_lvl_2

        # creating dataset for ranking
        df_match_candidates = pd.DataFrame(df[self._CONSTANTS['USER_COL']].unique())
        df_match_candidates.columns = [self._CONSTANTS['USER_COL']]
        df_match_candidates = df_match_candidates[
            df_match_candidates[self._CONSTANTS['USER_COL']].isin(
                dataset.data_train_lvl_1[self._CONSTANTS['USER_COL']].unique())]
        df_match_candidates['candidates'] = df_match_candidates[self._CONSTANTS['USER_COL']].apply(
            lambda x: self.get_own_recommendations(x, N=self._CONSTANTS['N_PREDICT']))

        df_items = df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1)\
                                      .stack()\
                                      .reset_index(level=1, drop=True)
        df_items.name = self._CONSTANTS['ITEM_COL']
        df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)

        # Создаем трейн сет для ранжирования с учетом кандидатов с этапа 1
        df_ranker_train = df[[self._CONSTANTS['USER_COL'], self._CONSTANTS['ITEM_COL']]].copy()
        df_ranker_train['target'] = 1  # тут только покупки
        df_ranker_train = df_match_candidates.merge(df_ranker_train,
                                                    on=[self._CONSTANTS['USER_COL'], self._CONSTANTS['ITEM_COL']],
                                                    how='left')
        df_ranker_train['target'].fillna(0, inplace=True)

        # merging
        df_ranker_train = df_ranker_train.merge(dataset.item_features, on=self._CONSTANTS['ITEM_COL'], how='left')
        df_ranker_train = df_ranker_train.merge(dataset.user_features, on=self._CONSTANTS['USER_COL'], how='left')

        if t == "train":
            # train split
            self.X_train = df_ranker_train.drop('target', axis=1)
            self.y_train = df_ranker_train[['target']]
        else:
            # test split
            self.X_test = df_ranker_train.drop('target', axis=1)
            self.y_test = df_ranker_train[['target']]

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=True)]
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user_id, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases[self._CONSTANTS['USER_COL']] == user_id].head(N)

        res = top_users_purchases[self._CONSTANTS['ITEM_COL']].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user_id, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user_id], N=N + 1)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for _user_id in similar_users:
            res.extend(self.get_own_recommendations(_user_id, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def ranker_fit(self):

        # формируем фичи для обучения
        self.cat_feats = self.X_train.columns[2:].tolist()

        train_pool = Pool(data=self.X_train[self.cat_feats],
                          label=self.y_train,
                          group_id=self.X_train[self._CONSTANTS['USER_COL']])

        test_pool = Pool(data=self.X_test[self.cat_feats],
                         label=self.y_test,
                         group_id=self.X_test[self._CONSTANTS['USER_COL']])

        parameters = {'loss_function': self._CONSTANTS['CATBOOST_RANKER'],
                      'train_dir': self._CONSTANTS['CATBOOST_RANKER'],
                      'iterations': self._CONSTANTS['CATBOOST_ITERATIONS'],
                      'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
                      'verbose': False,
                      'random_seed': 0,
                      }

        ranker_model = CatBoostRanker(**parameters)
        ranker_model.fit(train_pool, eval_set=test_pool, plot=True)
        self.ranker_model = ranker_model

    def ranker_predict(self, df):
        df['proba'] = catboost.CatBoost.predict(self.ranker_model,
                                                df,
                                                prediction_type='Probability')[:, 1]
        return df

    @staticmethod
    def compare_1lvl_models(self, result_lvl_1):
        recommenders = [name for name, val in MainRecommender.__dict__.items() if callable(val)][-4:]
        n = 50
        for r in recommenders:
            model_name_col = r.replace('get_', '').replace('_recommendations', '')
            result_lvl_1[model_name_col] = result_lvl_1[self._CONSTANTS['USER_COL']].apply(
                lambda x: eval(f'recommender.{r}({x}, N={n})'))
            result_lvl_1[model_name_col + '_score'] = result_lvl_1.apply(
                lambda x: precision_at_k(x[model_name_col], x['actual'], k=n), axis=1).mean()
            print(model_name_col, 'is ready.')

        score_columns = [item for item in result_lvl_1.columns.tolist() if 'score' in item]
        print(result_lvl_1[score_columns].head(1))
