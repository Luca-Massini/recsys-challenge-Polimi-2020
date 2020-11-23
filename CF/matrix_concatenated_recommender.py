from Recommender import Recommender
from utils.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.data_manager.data_manager import data_manager
import scipy.sparse as sparse
import numpy as np


class matrix_concatenated_recommender(Recommender):
    def __init__(self, icm=None):
        self.__urm_training = None
        if icm is None:
            self.__ICM = data_manager().get_icm()
        else:
            self.__ICM = icm
        self.__similarity_matrix = None

    def fit(self, training_set, k=100, shrink=100, normalize=True, similarity='cosine'):
        self.__urm_training = training_set
        new_icm = sparse.vstack([self.__ICM.T, self.__urm_training])
        print(new_icm.shape)
        similarity_object = Compute_Similarity_Python(new_icm,
                                                      shrink=shrink,
                                                      topK=k,
                                                      normalize=normalize,
                                                      similarity=similarity)
        self.__similarity_matrix = similarity_object.compute_similarity()
        print("similarity mtrix shape", self.__similarity_matrix.shape)

    def recommend(self, userId, at=10):
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
