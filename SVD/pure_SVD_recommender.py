from Recommender import Recommender
from utils.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
import numpy as np

SEED = 1234


class pure_SVD_recommender(Recommender):
    def __init__(self):
        self.__training_set = None
        self.__learner = None
        self.__similarity = None

    def fit(self, training_set, num_factors=40, k=100):
        self.__training_set = training_set
        self.__learner = PureSVDItemRecommender(URM_train=self.__training_set.copy())
        self.__similarity = self.__learner.fit(num_factors=num_factors, topK=k, random_seed=SEED)

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
