from Recommender import Recommender
from utils.GraphBased.P3alphaRecommender import P3alphaRecommender
import numpy as np
import scipy.sparse as sparse


class graph_based_recommender_alpha(Recommender):
    def __init__(self):
        self.__learner = None
        self.__similarity = None
        self.__urm_training = None
        self.__n_users = None
        self.__n_items = None

    def fit(self, training_set, k=100, alpha=1.):
        self.__n_users = training_set.shape[0]
        self.__n_items = training_set.shape[1]
        self.__urm_training = training_set
        self.__learner = P3alphaRecommender(URM_train=training_set)
        self.__similarity = self.__learner.fit(topK=k, alpha=alpha, normalize_similarity=False)

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

    def get_scores(self, userId):
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        return scores

    def compute_new_urm(self, ratings_per_user):
        rows = []
        for user in range(0, self.__n_users):
            if user+1 % 1000 == 0 and user != 0:
                print(user)
            recommended_items = self.recommend(user, at=ratings_per_user)
            row = np.zeros(shape=(self.__n_items,))
            for item in recommended_items:
                row[item] = 1
            rows.append(sparse.csr_matrix(sparse.coo_matrix(row)))
        new_urm = sparse.vstack(rows, format='csr')
        print("the new URM matrix is ready. Shape: ", new_urm.shape)
        return new_urm

