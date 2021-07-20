import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
import os, sys




def display_components_in_2D_space(components_df, labels='category', marker='D'):
    groups = components_df.groupby(labels)
    # Plot
    fig, ax = plt.subplots(figsize=(12 ,8))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.component_1, group.component_2,
                marker='o', ms=6,
                linestyle='',
                alpha=0.7,
                label=name)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.xlabel('component_1')
    plt.ylabel('component_2')
    plt.show()


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    hit_rate = (flags.sum() > 0) * 1
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    # your_code
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list)
    hit_rate = (flags.sum() > 0) * 1
    return hit_rate

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)
    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    # your_code
    # Лучше считать через скалярное произведение, а не цикл
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    precision = np.dot(prices_recommended, flags).sum() / prices_recommended.sum()
    return precision

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    # your_code
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(recommended_list, bought_list)
    recall = flags.sum() / len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    # your_code
    bought_list = np.array(bought_list)
    prices_bought = np.array(prices_bought)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(recommended_list, bought_list)
    recall = np.dot(prices_recommended, flags).sum() / prices_bought.sum()
    return recall


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(1, k + 1):

        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k

    result = sum_ / k

    return result

def map_k(recommended_lists, bought_lists, k=5):
    # your_code
    sum_ = 0
    for lists in zip(recommended_lists, bought_lists):
        bought_list = np.array(lists[1])
        recommended_list = np.array(lists[0])
        sum_ = 0
        sum_ += ap_k(recommended_list, bought_list, k=i)
    result = sum_ / len(bought_lists)
    return result

def reciprocal_rank(recommended_list, bought_list, k=5):
    # your_code
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[:k]
    flags = np.isin(recommended_list, bought_list)
    if sum(flags) == 0:
        return 0
    result = 1 / (np.where(flags == True)[0][0]+1)
    return result