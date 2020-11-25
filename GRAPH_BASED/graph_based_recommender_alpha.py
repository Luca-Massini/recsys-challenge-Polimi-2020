from Recommender import Recommender
from utils.GraphBased.P3alphaRecommender import P3alphaRecommender
import numpy as np


class graph_based_recommender_alpha(Recommender):
    def __init__(self):
        self.__learner = None
        self.__similarity = None
        self.__urm_training = None

    def fit(self, training_set, k=100, alpha=1.):
        self.__urm_training = training_set
        self.__learner = P3alphaRecommender(URM_train=training_set)
        self.__similarity = self.__learner.fit(topK=k, alpha=alpha, normalize_similarity=True)

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
