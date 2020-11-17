from Recommender import Recommender
from utils.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np


class cf_recommender_item(Recommender):

    def __init__(self):
        self.__URM = None
        self.__W_sparse = None

    def fit(self, training_set, k=100, shrink=100, normalize=True, similarity='cosine'):
        self.__URM = training_set
        similarity_object = Compute_Similarity_Python(self.__URM, shrink=shrink,
                                                      topK=k, normalize=normalize,
                                                      similarity=similarity)
        self.__W_sparse = similarity_object.compute_similarity()

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__URM[userId]
        scores = user_profile.dot(self.__W_sparse).toarray().ravel()
        if exclude_seen:
            scores = self.__filter_seen(userId, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__URM.indptr[user_id]
        end_pos = self.__URM.indptr[user_id + 1]
        user_profile = self.__URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores