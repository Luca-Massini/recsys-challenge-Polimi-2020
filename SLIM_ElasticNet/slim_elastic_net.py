from Recommender import Recommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import numpy as np


class slim_elastic_net(Recommender):
    def __init__(self):
        self.__training_set = None
        self.__learner = None
        self.__similarity = None

    def fit(self, training_set, l1_ratio=1e-5, alpha=1e-5, positive_only=True, k=100):
        self.__training_set = training_set
        self.__learner = SLIMElasticNetRecommender(URM_train=self.__training_set)
        self.__similarity = self.__learner.fit(l1_ratio=l1_ratio,
                                               alpha=alpha,
                                               positive_only=positive_only,
                                               topK=k)

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__training_set[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        scores = self.__filter_seen(userId, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__training_set.indptr[user_id]
        end_pos = self.__training_set.indptr[user_id + 1]
        user_profile = self.__training_set.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def get_similarity_matrix(self):
        return self.__similarity
