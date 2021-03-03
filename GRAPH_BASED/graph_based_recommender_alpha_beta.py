from Recommender import Recommender
from utils.GraphBased.RP3betaRecommender import RP3betaRecommender
import numpy as np


class graph_based_recommender_alpha_beta(Recommender):
    def __init__(self):
        self.__urm_training = None
        self.__learner = None
        self.__similarity = None

    def fit(self, training_set, alpha=1., beta=1., k=100):
        self.__urm_training = training_set
        self.__learner = RP3betaRecommender(URM_train=training_set)
        self.__similarity = self.__learner.fit(alpha=alpha,
                                               beta=beta,
                                               min_rating=0,
                                               topK=k, implicit=True, normalize_similarity=True)

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
