from Recommender import Recommender
from basic_recommenders.TopPopRecommender import TopPopRecommender
from utils.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.data_manager.data_manager import data_manager
import scipy.sparse as sparse
import numpy as np


class item_knn_cf_cbf(Recommender):
    def __init__(self, icm=None):
        self.__urm_training = None
        if icm is None:
            self.__ICM = data_manager().get_icm()
        else:
            self.__ICM = icm
        self.__similarity_matrix = None
        self.__non_personalized_recommender = TopPopRecommender()
        self.__n_users = None
        self.__n_items = None

    def fit(self, training_set, k=100, shrink=100, normalize=True, similarity='cosine'):
        self.__n_items = training_set.shape[1]
        self.__n_users = training_set.shape[0]
        self.__urm_training = training_set
        self.__non_personalized_recommender.fit(training_set=training_set)
        if self.__ICM is not None:
            new_icm = sparse.vstack([self.__ICM.T, self.__urm_training])
        else:
            new_icm = self.__urm_training
        similarity_object = Compute_Similarity_Python(new_icm,
                                                      shrink=shrink,
                                                      topK=k,
                                                      normalize=normalize,
                                                      similarity=similarity)
        self.__similarity_matrix = similarity_object.compute_similarity()

    def recommend(self, userId, at=10):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        if user_profile.nnz == 0:
            return self.__non_personalized_recommender.recommend(userId=userId, at=at)
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

    def get_similarity_matrix(self):
        return self.__similarity_matrix

    def get_scores(self, userId, filter_seen=True):
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity_matrix).toarray().ravel()
        return scores

    def compute_new_urm(self, ratings_per_user):
        print("computing the new urm matrix")
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
