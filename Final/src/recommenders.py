import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
import itertools


def reduce_dims(df, dims=2, method='pca'):
    assert method in ['pca', 'tsne'], 'Неверно указан метод'
    if method == 'pca':
        pca = PCA(n_components=dims)
        components = pca.fit_transform(df)
    elif method == 'tsne':
        tsne = TSNE(n_components=dims, learning_rate=250, random_state=42)
        components = tsne.fit_transform(df)
    else:
        print('Error')
    colnames = ['component_' + str(i) for i in range(1, dims + 1)]
    return pd.DataFrame(data=components, columns=colnames)


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='sales_value',
                                          aggfunc='count',
                                          fill_value=0)
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
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

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        self.
        rec_results['als'] = rec_results['user_id'].apply(lambda x: get_recommendations(x, model=model, N=N))
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        # your_code
        res = [id_to_itemid[rec[0]] for rec in self.model.recommend(userid=userid_to_id[user],
                                                                    user_items=sparse_user_item,
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=None,
                                                                    recalculate_user=True)]
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_recommendations(user, model, N=5):
        return [id_to_itemid[rec[0]] for rec in model.recommend(userid=userid_to_id[user],
                                                                user_items=sparse_user_item,
                                                                N=N,
                                                                filter_already_liked_items=False,
                                                                filter_items=None,
                                                                recalculate_user=True)]

    def als_grid_search_cv(user_item_matrix, rec_results, params, N=5):
        scores = []
        for factors, reg, iters in itertools.product(*params.values()):
            model = AlternatingLeastSquares(factors=factors,
                                            regularization=reg,
                                            iterations=iters,
                                            calculate_training_loss=True,
                                            num_threads=6)
            model.fit(csr_matrix(user_item_matrix).T, show_progress=False)

            rec_results['als'] = rec_results['user_id'].apply(lambda x: get_recommendations(x, model=model, N=N))
            precision = rec_results.apply(lambda row: precision_at_k(row['als'], row['actual']), axis=1).mean()
            score = {'factors': factors,
                     'regularization': reg,
                     'iterations': iters,
                     'precision': precision}
            scores.append(score)
        return scores

    def best_params(models_scores, all=False):
        df = pd.DataFrame(models_scores, index=range(len(models_scores))).sort_values('precision', ascending=False)
        if all:
            return df
        df = df.head(1)
        return df.to_dict()
