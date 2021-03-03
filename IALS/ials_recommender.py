from Recommender import Recommender
from utils.MatrixFactorization.IALSRecommender import IALSRecommender
import numpy as np


class ials_recommender(Recommender):
    def __init__(self):
        self.__urm_training = None
        self.__learner = None
        self.__ratings = None

    def fit(self, training_set,
            epochs=5,
            num_factors=300,
            alpha=24,
            epsilon=1.0,
            reg=1e-2):
        self.__urm_training = training_set
        self.__learner = IALSRecommender(URM_train=training_set)
        user_factors, item_factors = self.__learner.fit(epochs=epochs,
                                                        num_factors=num_factors,
                                                        alpha=alpha,
                                                        epsilon=epsilon,
                                                        reg=reg)
        self.__ratings = user_factors.dot(item_factors.T)
        print("rating shape: ", self.__ratings.shape)

    def recommend(self, userId, at):
        user_scores = self.__ratings[userId, :].ravel()
        scores = self.__filter_seen(user_id=userId, scores=user_scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]

        user_profile = self.__urm_training.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
