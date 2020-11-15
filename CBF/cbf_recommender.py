import numpy as np
from Recommender import trained_Recommender
from utils.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.data_manager.data_manager import data_manager
from utils.data_manager.splitter import splitter
from utils.evaluator.evaluator import evaluator


class cbf_recommender(trained_Recommender):
    def __init__(self):
        self.__training_set = None
        self.__ICM = data_manager().get_icm()
        self.__similarity_matrix = None

    def fit(self, training_set, k=100, shrink=100, normalize=True, similarity='cosine'):
        self.__training_set = training_set
        similarity_object = Compute_Similarity_Python(self.__ICM.T, shrink=shrink,
                                                      topK=k, normalize=normalize,
                                                      similarity=similarity)

        self.__similarity_matrix = similarity_object.compute_similarity()

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__training_set[userId]
        scores = user_profile.dot(self.__similarity_matrix).toarray().ravel()

        if exclude_seen:
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


if __name__ == '__main__':
    recommender = cbf_recommender()
    map_ = evaluator(recommender_object=recommender).evaluate()
    print(map_)
