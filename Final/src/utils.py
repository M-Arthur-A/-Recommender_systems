import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
import os, sys


def prefilter_items(data, irrelevant=None, price_thresholds=(0, 1000000000)):
    """
    irrelevant - list of irrelevant departments of products to remove from data
    price_thresholds[0] - threshold of cheap items (min price)
    price_thresholds[1] - threshold of expensive items (max price)
    """
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    max_day = data['day'].max()
    items_365 = data.loc[(data['day'] <= max_day) & (data['day'] >= max_day - 365), 'item_id'].unique().tolist()
    data = data.loc[data['item_id'].isin(items_365)]

    # Уберем не интересные для рекоммендаций категории (department)
    if irrelevant:
        products = pd.DataFrame(r'../_ADDS/webinar_3/data/product.csv')
        relevant_products = products.loc[~products['DEPARTMENT'].isin(irrelevant), 'PRODUCT_ID'].inique().tolist()
        data = data.loc[data['item_id'].isin(relevant_products)]

    # Уберем слишком дешевые / дорогие товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    if price_thresholds[0] or price_thresholds[1] != 1000000000:
        data = data.loc[(data['price'] >= price_thresholds[0]) & (data['price'] <= price_thresholds[1])]

    # ...
    return data


def postfilter_items(user_id, recommednations):
    pass

