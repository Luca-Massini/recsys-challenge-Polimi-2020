from Recommender import Recommender
from utils.SLIM_BPR.SLIM_BPR import SLIM_BPR
import numpy as np
import scipy.sparse as sparse

from utils.data_manager.data_manager import data_manager


class slim_bpr(Recommender):
    def __init__(self):
        self.__training_set = None
        self.__learner = None
        self.__similarity = None
        self.__icm = None

    def fit(self, training_set, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05, k=100, epochs=20, concatenate_icm=False):
        self.__training_set = training_set
        if concatenate_icm:
            self.__icm = data_manager().get_icm()
            new_urm = sparse.vstack([self.__icm.T, self.__training_set], format='csr')
        else:
            new_urm = self.__training_set
        self.__learner = SLIM_BPR(URM_train=new_urm, lambda_i=lambda_i,
                                  lambda_j=lambda_j, learning_rate=learning_rate)
        self.__similarity = self.__learner.fit(epochs=epochs, k=k)

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
