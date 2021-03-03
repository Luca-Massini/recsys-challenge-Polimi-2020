from Recommender import Recommender
from utils.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from utils.Recommender_utils import similarityMatrixTopK
from utils.data_manager.data_manager import data_manager
import numpy as np
from scipy.sparse.linalg import svds
import scipy.sparse as sparse


class pure_SVD_icm(Recommender):
    def __init__(self):
        self.__urm_training = None
        self.__icm = None
        self.__similarity = None
        self.__learner = None

    def fit(self, training_set, num_factors=100, k=100):
        self.__urm_training = training_set
        self.__icm = data_manager().get_icm()
        print("icm shape: ", self.__icm.shape)
        self.__learner = PureSVDItemRecommender(URM_train=self.__icm.T.copy())
        self.__similarity = self.__learner.fit(num_factors=num_factors,
                                               topK=k)
        print("shape of the similarity matrix: ", self.__similarity.shape)

    def recommend(self, userId, at):
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
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
