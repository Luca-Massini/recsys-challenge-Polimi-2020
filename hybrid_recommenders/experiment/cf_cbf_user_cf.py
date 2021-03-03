from GRAPH_BASED.graph_based_recommender_alpha import graph_based_recommender_alpha
from Recommender import Recommender
from SLIM_ElasticNet.slim_elastic_net import slim_elastic_net
from hybrid_recommenders.experiment.combined_user_cf import combined_user_cf
from hybrid_recommenders.item_knn_cf_cbf.item_knn_cf_cbf_different_similarity import \
    item_knn_cf_cbf_different_similarity
from sklearn.preprocessing import normalize
import numpy as np


class cf_cbf_user_cf(Recommender):
    def __init__(self, weight=None, normalization=None):
        if normalization is None:
            normalization = [1, 0, 1, 1]
        self.__normalization = normalization
        if weight is None:
            weight = [0.1]
        self.__cf_cbf_similarities = None
        self.__cf_user_learner = None
        self.__graph_learner = None
        self.__slim_learner = None
        self.__urm_training = None
        self.__cf_user_similarity = None
        self.__cf_cbf_item_similarity = None
        self.__graph_similarity = None
        self.__slim_similarity = None
        self.__weight = weight
        self.__new_urm = None
        self.__similarity_matrices = []

    def fit(self, training_set, k_user_cf=100, shrink_user_cf=20, similarity_user_cf='cosine', weights_cf_cbf=None,
            similarity_1_cf_cbf='cosine', similarity_2_cf_cbf='dice', k1_cf_cbf=150, k2_cf_cbf=70,
            shrink_1_cf_cbf=450, shrink_2_cbf_cbf=246):
        if weights_cf_cbf is None:
            weights_cf_cbf = [0.5, 0.5]
        self.__urm_training = training_set
        self.__cf_user_learner = combined_user_cf()
        self.__cf_user_learner.fit(training_set=training_set,
                                   k=k_user_cf,
                                   shrink=shrink_user_cf,
                                   similarity=similarity_user_cf,
                                   rating_per_user=3)
        self.__new_urm = self.__cf_user_learner.get_enriched_urm()
        self.__cf_user_similarity = self.__cf_user_learner.get_similarity_matrix()
        self.__similarity_matrices.append(self.__cf_user_similarity.copy())
        self.__cf_cbf_similarities = item_knn_cf_cbf_different_similarity(similarity1=similarity_1_cf_cbf,
                                                                          similarity2=similarity_2_cf_cbf,
                                                                          weight1=weights_cf_cbf[0],
                                                                          weight2=weights_cf_cbf[1],
                                                                          )
        self.__cf_cbf_similarities.fit(training_set=training_set.copy(),
                                       k1=k1_cf_cbf,
                                       k2=k2_cf_cbf,
                                       shrink1=shrink_1_cf_cbf,
                                       shrink2=shrink_2_cbf_cbf)
        self.__cf_cbf_item_similarity = self.__cf_cbf_similarities.get_similarity_matrix()
        self.__similarity_matrices.append(self.__cf_cbf_item_similarity.copy())
        self.__graph_learner = graph_based_recommender_alpha()
        self.__graph_learner.fit(training_set=self.__new_urm,
                                 alpha=0.5,
                                 k=200)
        self.__graph_similarity = self.__graph_learner.get_similarity_matrix()
        self.__similarity_matrices.append(self.__graph_similarity.copy())
        self.__slim_learner = slim_elastic_net(use_urm_augmentation=False,
                                               n_ones=3,
                                               n_scores=0)
        self.__slim_learner.fit(training_set=self.__new_urm,
                                l1_regularization=5e-3,
                                l2_regularization=0.005,
                                k=300,
                                use_normalization=False,
                                normalization_type='l2')
        self.__slim_similarity = self.__slim_learner.get_similarity_matrix()
        self.__similarity_matrices.append(self.__slim_similarity.copy())
        self.__cf_user_similarity, self.__cf_cbf_item_similarity, self.__graph_similarity, self.__slim_similarity = self.__normalize_matrices(self.__similarity_matrices)

    def recommend(self, userId, at):
        user_profile = self.__urm_training[userId]
        scores_cf_cbf = user_profile.dot(self.__cf_cbf_item_similarity).toarray().ravel()
        scores_user_cf = self.__cf_user_similarity[userId, :].dot(self.__urm_training).toarray().ravel()
        scores_graph = user_profile.dot(self.__graph_similarity).toarray().ravel()
        scores_slim = user_profile.dot(self.__slim_similarity).toarray().ravel()
        scores = scores_user_cf * self.__weight[0] + scores_cf_cbf * self.__weight[1] + scores_graph * self.__weight[
            2] + scores_slim * self.__weight[3]
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]
        user_profile = self.__urm_training.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def change_weight(self, new_weight):
        self.__weight = new_weight

    def __normalize_matrices(self, similarity_matrices):
        normalized_matrices = []
        for i in range(0, len(similarity_matrices)):
            if self.__normalization[i] == 1:
                if i != 0:
                    normalized_matrices.append(normalize(similarity_matrices[i], norm='l2', axis=1))
                else:
                    normalized_matrices.append(normalize(similarity_matrices[i], norm='l2', axis=0))
            else:
                normalized_matrices.append(similarity_matrices[i])
        return normalized_matrices

    def change_normalization(self, new_normalization_vector):
        self.__normalization = new_normalization_vector
        self.__cf_user_similarity, self.__cf_cbf_item_similarity, self.__graph_similarity, self.__slim_similarity = self.__normalize_matrices(
            self.__similarity_matrices)
