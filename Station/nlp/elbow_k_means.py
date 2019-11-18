from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.elbow import elbow
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils.metric import distance_metric, type_metric
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from .ModelSingleton import *


def elbow_k_means(key_word, model_path):
    logger = Logger(model_path)
    model = logger.model
    result = model.most_similar(key_word, topn=100)

    word_vectors = []
    num_clusters = 8
    word_names = []
    word_correlation = []
    for r in result:
        word_vectors.append(model.wv[r[0]])
        word_names.append(r[0])
        word_correlation.append(r[1])

    tsne = PCA(n_components=2)

    X_tsne = tsne.fit_transform(word_vectors)

    kmin, kmax = 1, 10
    elbow_instance = elbow(X_tsne, kmin, kmax)

    elbow_instance.process()
    amount_clusters = elbow_instance.get_amount()
    wce = elbow_instance.get_wce()

    centers = kmeans_plusplus_initializer(X_tsne,
                                          amount_clusters,
                                          amount_candidates=kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
    k_means_instance = kmeans(X_tsne, centers)
    k_means_instance.process()

    clusters = k_means_instance.get_clusters()
    centers = k_means_instance.get_centers()

    index_to_word = [[] for i in range(len(clusters))]
    index_to_correlation = [[] for i in range(len(clusters))]
    idx = 0
    cluster_list = []
    for c in clusters:
        words_list = []
        for i in c:
            word_dict = dict()
            word_dict["text"] = word_names[i]
            word_dict["correlation"] = word_correlation[i]
            t_dict = dict()
            t_dict["word"] = word_dict
            words_list.append(t_dict)
        words_dict = dict()
        words_dict["words"] = words_list
        cluster_list.append(words_dict)
        idx += 1

    return len(clusters), cluster_list
