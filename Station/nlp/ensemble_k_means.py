from sklearn.cluster import KMeans
from munkres import Munkres
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def doing_kmeans(data, n_clusters):
    KM = KMeans(n_clusters=n_clusters)
    KM.fit(data)
    return KM.labels_


def translate_clustering(clt, mapper):
    return np.array([mapper[i] for i in clt])


# hungarian algorithm.
def make_cost_matrix(c1, c2):
    x1 = np.unique(c1)
    x2 = np.unique(c2)
    y1 = x1.size
    y2 = x2.size
    assert(y1 == y2 and np.all(x1 == x2))

    P = np.ones([y1, y2])
    for i in range(y1):
        it_i = np.nonzero(c1 == x1[i])[0]
        for j in range(y2):
            it_j = np.nonzero(c2 == x2[j])[0]
            P_ij = np.intersect1d(it_j, it_i)
            P[i, j] = -P_ij.size
    return P


def generate_kmeans(data, n_clusters, n_ensembles):
    labels = {}
    for i in range(n_ensembles):
        labels[i] = doing_kmeans(data, n_clusters)
    return labels


# Arranging the result of sklearn kmeans labels which have different result for each Kmeans algorithms
# This is hungarian algorithm
def arranging_kmeans(labels):
    m = Munkres()
    # make cost matrix
    new_labels = {}
    for i in range(len(labels.keys())):
        if i == 0:
            new_labels[i] = labels[i]
        else:
            cost_matrix = make_cost_matrix(labels[i], labels[0])
            indexes = m.compute(cost_matrix)
            mapper = {old:new for (old, new) in indexes}
            new_labels = translate_clustering(labels[i], mapper)
    return new_labels


def voting(series):
    val = series.value_counts().sort_values(ascending=False)
    return val.index[0]


def voting_labels(dict_labels):
    labels = pd.DataFrame.from_dict(dict_labels)
    labels = labels.apply(lambda x:voting(x), axis=1)
    return labels


def ensemble_kmeans(data, n_clusters, n_ensembles):
    generate_kmeans_labels = generate_kmeans(data=data, n_clusters=n_clusters, n_ensembles=n_ensembles)
    hungarian = arranging_kmeans(generate_kmeans_labels)
    voted_labels = voting_labels(hungarian)

    return voted_labels


def generate_word(key_word, model_path):
    model = Word2Vec.load(model_path)
    result = model.most_similar(key_word, topn=90)
    word_vectors = []
    word_names = []
    for r in result:
        word_vectors.append(model.wv[r[0]])
        word_names.append(r[0])

    return word_vectors, word_names


def get_cluster(key_word, model_path, n_clusters=8, n_ensembles=30):
    word_vectors, word_names = generate_word(key_word=key_word, model_path=model_path)

    result = ensemble_kmeans(word_vectors, n_clusters=n_clusters, n_ensembles=n_ensembles)

    cluster = [list() for _ in range(n_clusters)]

    for i in range(len(result)):
        cluster[result[i]].append(word_names[i])

    return cluster