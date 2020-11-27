from CF.cf_recommender_user import cf_recommender_user
from GRAPH_BASED.graph_based_recommender_alpha import graph_based_recommender_alpha
from Recommender import Recommender
from hybrid_recommenders.item_knn_cf_cbf.item_knn_cf_cbf_concatenated import item_knn_cf_cbf
import numpy as np


class graph_item_cf_cbf_user_cf(Recommender):

    def __init__(self, weights):
        assert (len(weights) == 4)
        self.__urm_training = None
        self.__learner1 = None
        self.__learner2 = None
        self.__learner3 = None
        self.__learner4 = None
        self.__similarity_1 = None
        self.__similarity_2 = None
        self.__similarity_3 = None
        self.__similarity_4 = None
        self.__similarity = None
        self.__weights = weights

    def fit(self, training_set, k1_cf_cbf=100, shrink1_cf_cbf=100, normalize=True, similarity1_cf_cbf='cosine',
            k2_cf_cbf=100, shrink2_cf_cbf=100, similarity2_cf_cbf='cosine', alpha=0.1, k_graph=100,
            similarity_user_cf='cosine', shrink_user_cf=100, k_user_cf=100):
        self.__urm_training = training_set
        self.__learner1 = item_knn_cf_cbf()
        self.__learner2 = item_knn_cf_cbf()
        self.__learner3 = graph_based_recommender_alpha()
        self.__learner4 = cf_recommender_user()
        self.__learner1.fit(training_set=training_set.copy(),
                            k=k1_cf_cbf,
                            shrink=shrink1_cf_cbf,
                            normalize=True,
                            similarity=similarity1_cf_cbf)
        self.__similarity_1 = self.__learner1.get_similarity_matrix()
        self.__similarity_1 = self.__similarity_1 / self.__similarity_1.max()
        self.__learner2.fit(training_set=training_set.copy(),
                            k=k2_cf_cbf,
                            shrink=shrink2_cf_cbf,
                            normalize=True,
                            similarity=similarity2_cf_cbf)
        self.__similarity_2 = self.__learner2.get_similarity_matrix()
        self.__similarity_2 = self.__similarity_2 / self.__similarity_2.max()
        self.__learner3.fit(training_set.copy(),
                            k=k_graph,
                            alpha=alpha)
        self.__similarity_3 = self.__learner3.get_similarity_matrix()
        self.__similarity_3 = self.__similarity_3 / self.__similarity_3.max()
        self.__learner4.fit(training_set.copy(),
                            k=k_user_cf,
                            shrink=shrink_user_cf,
                            normalize=True,
                            similarity=similarity_user_cf)
        self.__similarity_4 = self.__learner4.get_similarity_matrix()
        self.__similarity_4 = self.__similarity_4 / self.__similarity_4.max()
        self.__similarity = self.__similarity_1 * self.__weights[0] + self.__similarity_2 * self.__weights[1] \
                            + self.__similarity_3 * self.__weights[2]

    def recommend(self, userId, at=10):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        weighted_user_cf_scores = (self.__similarity_4[userId, :]*self.__weights[3]).dot(self.__urm_training).toarray().ravel()
        scores = user_profile.dot(self.__similarity).toarray().ravel() + weighted_user_cf_scores
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]
        user_profile = self.__urm_training.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores
