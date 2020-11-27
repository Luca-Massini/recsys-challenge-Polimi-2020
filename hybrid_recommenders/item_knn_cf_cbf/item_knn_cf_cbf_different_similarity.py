from Recommender import Recommender
#from hybrid_recommenders.item_knn_cf_cbf_concatenated import item_knn_cf_cbf
import numpy as np

from hybrid_recommenders.item_knn_cf_cbf.item_knn_cf_cbf_concatenated import item_knn_cf_cbf


class item_knn_cf_cbf_different_similarity(Recommender):
    def __init__(self, similarity1, similarity2, weight1, weight2):
        self.__learner_1 = None
        self.__learner_2 = None

        self.__urm_training = None
        self.__similarity_m_1 = None
        self.__similarity_m_2 = None

        self.__similarity = None
        self.__similarity1 = similarity1
        self.__similarity2 = similarity2

        self.__weight1 = weight1
        self.__weight2 = weight2

    def fit(self, training_set, k1=100, k2=100, shrink1=20, shrink2=20):
        self.__urm_training = training_set
        self.__learner_1 = item_knn_cf_cbf()
        self.__learner_2 = item_knn_cf_cbf()
        self.__learner_1.fit(training_set,
                             k=k1,
                             shrink=shrink1,
                             normalize=True,
                             similarity=self.__similarity1)
        self.__learner_2.fit(training_set,
                             k=k2,
                             shrink=shrink2,
                             normalize=True,
                             similarity=self.__similarity2)
        self.__similarity_m_1 = self.__learner_1.get_similarity_matrix()
        self.__similarity_m_2 = self.__learner_2.get_similarity_matrix()

        self.__similarity_m_1 = self.__similarity_m_1 / self.__similarity_m_1.max()
        self.__similarity_m_2 = self.__similarity_m_2 / self.__similarity_m_2.max()

        self.__similarity = self.__weight1 * self.__similarity_m_1 + self.__weight2 * self.__similarity_m_2

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

    def get_similarity_matrix(self):
        return self.__similarity
