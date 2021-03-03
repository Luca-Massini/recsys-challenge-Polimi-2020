from CBF.cbf_recommender import cbf_recommender
from Recommender import Recommender
import numpy as np

class cbf_recommender_different_similarities(Recommender):
    def __init__(self, weights):
        assert (len(weights) == 2)
        self.__urm_training = None
        self.__learner1 = None
        self.__learner2 = None
        self.__weights = weights
        self.__similarity1 = None
        self.__similarity2 = None
        self.__similarity = None

    def fit(self, training_set, k1=100, shrink1=100, normalize=True, similarity1='cosine',
            k2=100, shrink2=100, similarity2='cosine'):
        self.__urm_training = training_set
        self.__learner1 = cbf_recommender()
        self.__learner2 = cbf_recommender()
        self.__learner1.fit(training_set=training_set.copy(),
                            k=k1,
                            shrink=shrink1,
                            normalize=False,
                            similarity=similarity1)
        self.__similarity1 = self.__learner1.get_similarity_matrix()
        self.__learner2.fit(training_set=training_set.copy(),
                            k=k2,
                            shrink=shrink2,
                            normalize=False,
                            similarity=similarity2)
        self.__similarity2 = self.__learner2.get_similarity_matrix()
        if normalize:
            self.__similarity2 = self.__similarity2 / self.__similarity2.max()
            self.__similarity1 = self.__similarity1 / self.__similarity1.max()
        self.__similarity = self.__similarity1 * self.__weights[0] + self.__similarity2 * self.__weights[1]

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()

        if exclude_seen:
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

    def get_similarity_matrix(self):
        return self.__similarity
