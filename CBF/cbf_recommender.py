import numpy as np
from Recommender import Recommender
from utils.Similarity.Compute_Similarity_Euclidean import Compute_Similarity_Euclidean
from utils.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.data_manager.data_manager import data_manager
import scipy.sparse as sparse
from utils.evaluator.evaluator import evaluator


class cbf_recommender(Recommender):
    def __init__(self, icm=None):
        self.__n_users = None
        self.__n_items = None
        self.__urm_training = None
        if icm is None:
            self.__ICM = data_manager().get_icm()
        else:
            self.__ICM = icm
        self.__similarity_matrix = None

    def fit(self, training_set, k=100, shrink=100, normalize=True, similarity='cosine',similarity_from_distance_mode="lin"):
        self.__n_users = training_set.shape[0]
        self.__n_items = training_set.shape[1]
        self.__urm_training = training_set
        if similarity != 'euclidean':
            similarity_object = Compute_Similarity_Python(self.__ICM.T, shrink=shrink,
                                                          topK=k, normalize=normalize,
                                                          similarity=similarity)
            self.__similarity_matrix = similarity_object.compute_similarity()
        else:
            similarity_object = Compute_Similarity_Euclidean(dataMatrix=self.__ICM.T,
                                                             topK=k,
                                                             shrink=shrink,
                                                             normalize=normalize,
                                                             similarity_from_distance_mode=similarity_from_distance_mode)
            self.__similarity_matrix = similarity_object.compute_similarity()

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity_matrix).toarray().ravel()

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
        return self.__similarity_matrix

    def get_scores(self, userId):
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity_matrix).toarray().ravel()
        return scores

    def get_new_urm(self, ratings_per_user):
        rows = []
        print("ratings per user: ", ratings_per_user)
        for user in range(0, self.__n_users):
            recommended_items = self.recommend(user, at=ratings_per_user, exclude_seen=True)
            scores = self.get_scores(userId=user)
            row = np.zeros(shape=(self.__n_items,))
            for item in recommended_items:
                row[item] = scores[item]
            rows.append(sparse.csr_matrix(sparse.coo_matrix(row)))
        new_urm = sparse.vstack(rows, format='csr')
        print("the new URM matrix is ready. Shape: ", new_urm.shape)
        return new_urm

