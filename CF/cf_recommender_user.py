from Recommender import Recommender
from utils.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
import scipy.sparse as sparse


class cf_recommender_user(Recommender):
    def __init__(self):
        self.__URM = None
        self.__W_sparse = None
        self.__n_users = None
        self.__n_items = None

    def fit(self, training_set, k=100, shrink=100, normalize=False, similarity='cosine'):
        self.__URM = training_set
        self.__n_users = training_set.shape[0]
        self.__n_items = training_set.shape[1]
        similarity_object = Compute_Similarity_Python(self.__URM.T, shrink=shrink,
                                                      topK=k, normalize=normalize,
                                                      similarity=similarity)

        self.__W_sparse = similarity_object.compute_similarity()

    def recommend(self, userId, at=None, exclude_seen=True):
        # compute the scores using the dot product

        scores = self.__W_sparse[userId, :].dot(self.__URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(userId, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, userId):
        scores = self.__W_sparse[userId, :].dot(self.__URM).toarray().ravel()
        return scores

    def filter_seen(self, user_id, scores):
        start_pos = self.__URM.indptr[user_id]
        end_pos = self.__URM.indptr[user_id + 1]

        user_profile = self.__URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def get_new_urm(self, ratings_per_user):
        rows = []
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

    def get_similarity_matrix(self):
        return self.__W_sparse
