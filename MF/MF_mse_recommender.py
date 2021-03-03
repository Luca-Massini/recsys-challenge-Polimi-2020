from Recommender import Recommender
from utils.MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
import scipy.sparse as sparse
import numpy as np
from utils.evaluator.evaluator import evaluator


from utils.data_manager.splitter import splitter


class MF_mse_recommender(Recommender):
    def __init__(self):
        self.__urm_train = None
        self.__learner = None
        self.__user_factors = None
        self.__item_factors = None
        self.__ratings = None

    def fit(self, training_set, epochs=100, batch_size=100, num_factors=10,
            learning_rate=0.0005, use_cuda=False):
        self.__urm_train = training_set
        self.__learner = MF_MSE_PyTorch(URM_train=self.__urm_train.copy())
        self.__learner.fit(epochs=epochs, batch_size=batch_size, num_factors=num_factors,
                           learning_rate=learning_rate, use_cuda=use_cuda)
        self.__item_factors = sparse.csr_matrix(self.__learner.ITEM_factors)
        self.__user_factors = sparse.csr_matrix(self.__learner.USER_factors)
        self.__ratings = self.__user_factors.dot(self.__item_factors.T)

    def recommend(self, userId, at):
        user_scores = self.__ratings[userId, :].toarray().ravel()
        scores = self.__filter_seen(user_id=userId, scores=user_scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_train.indptr[user_id]
        end_pos = self.__urm_train.indptr[user_id + 1]

        user_profile = self.__urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
