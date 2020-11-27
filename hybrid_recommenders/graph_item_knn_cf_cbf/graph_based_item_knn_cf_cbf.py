from Recommender import Recommender
from GRAPH_BASED.graph_based_recommender_alpha import graph_based_recommender_alpha
import numpy as np

from hybrid_recommenders.item_knn_cf_cbf.item_knn_cf_cbf_different_similarity import \
    item_knn_cf_cbf_different_similarity
from utils.Recommender_utils import similarityMatrixTopK


class graph_based_item_item_knn_cf_cbf(Recommender):
    def __init__(self, weight1, weight2, ensemble_weight, first_similarity, second_similarity, k_tot=None):
        self.__similarity = None
        self.urm_training = None
        self.__learner1 = None
        self.__learner2 = None
        self.__urm_training = None
        self.__weight1 = weight1
        self.__weight2 = weight2
        self.__ensemble_weight = ensemble_weight
        self.__first_similarity = first_similarity
        self.__second_similarity = second_similarity
        self.__similarity1 = None
        self.__similarity2 = None
        self.__k_tot = k_tot

    def fit(self, training_set, k_graph=100, alpha=1., k1_cf_cbf=100, k2_cf_cbf=100, shrink1_cf_cbf=20,
            shrink2_cf_cbf=20):
        self.__urm_training = training_set
        self.__learner1 = graph_based_recommender_alpha()
        self.__learner1.fit(training_set=self.__urm_training,
                            k=k_graph,
                            alpha=alpha)
        self.__learner2 = item_knn_cf_cbf_different_similarity(similarity1=self.__first_similarity,
                                                               similarity2=self.__second_similarity,
                                                               weight1=self.__weight1,
                                                               weight2=self.__weight2)
        self.__learner2.fit(training_set=self.__urm_training,
                            k1=k1_cf_cbf,
                            k2=k2_cf_cbf,
                            shrink1=shrink1_cf_cbf,
                            shrink2=shrink2_cf_cbf)
        self.__similarity1 = self.__learner1.get_similarity_matrix()
        self.__similarity2 = self.__learner2.get_similarity_matrix()
        self.__similarity = self.__similarity1 * self.__ensemble_weight + self.__similarity2 * (
                    1 - self.__ensemble_weight)
        if self.__k_tot is not None:
            self.__similarity = similarityMatrixTopK(self.__similarity, k=self.__k_tot)

    def recommend(self, userId, at=10):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]
        user_profile = self.__urm_training.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores