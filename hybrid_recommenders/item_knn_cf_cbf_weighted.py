from CBF.cbf_recommender import cbf_recommender
from CF.cf_recommender_item import cf_recommender_item
from Recommender import Recommender
import numpy as np
from utils.Recommender_utils import similarityMatrixTopK


class item_knn_cf_cbf_weighted(Recommender):
    def __init__(self, weight_cf, weight_cbf, cf_similarity, cbf_similarity):
        #assert (weight_cbf + weight_cf == 1)
        self.__urm_training = None
        self.__similarity_matrix = None
        self.__cf_similarity = cf_similarity
        self.__cbf_similarity = cbf_similarity
        self.__weight_cf = weight_cf
        self.__weight_cbf = weight_cbf
        self.__cf = cf_recommender_item()
        self.__cbf = cbf_recommender()

    def fit(self, training_set, k_cf=100, k_cbf=100, shrink_cf=20, shrink_cbf=20, k_total=None):
        self.__urm_training = training_set
        self.__cf.fit(training_set=self.__urm_training, k=k_cf, shrink=shrink_cf, similarity=self.__cf_similarity)
        self.__cbf.fit(training_set=self.__urm_training, k=k_cf, shrink=shrink_cf, similarity=self.__cbf_similarity)
        self.__cf_similarity = self.__cf.get_similarity_matrix()
        self.__cbf_similarity = self.__cbf.get_similarity_matrix()
        self.__similarity_matrix = self.__weight_cf * self.__cf_similarity + self.__weight_cbf * self.__cbf_similarity
        if k_total is not None:
            self.__similarity_matrix = similarityMatrixTopK(self.__similarity_matrix.copy(), k=k_total)

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity_matrix).toarray().ravel()
        scores = self.__filter_seen(userId, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]

        user_profile = self.__urm_training.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def set_weights(self, weight_cf, weight_cbf):
        self.__weight_cf = weight_cf
        self.__weight_cbf = weight_cbf
        self.__similarity_matrix = self.__weight_cf * self.__cf_similarity + self.__weight_cbf * self.__cbf_similarity