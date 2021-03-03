from CBF.cbf_recommender import cbf_recommender
from Recommender import Recommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import  MultiThreadSLIM_ElasticNet
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize


class slim_elastic_net(Recommender):
    def __init__(self, use_urm_augmentation=False, n_ones=3, n_scores=4):
        self.__training_set = None
        self.__new_urm = None
        self.__learner = None
        self.__similarity = None
        self.__n_users = None
        self.__n_items = None
        self.__use_augmentation = use_urm_augmentation
        self.__n_ones = n_ones
        self.__n_scores = n_scores

    def fit(self, training_set, l1_regularization=1e-6, l2_regularization=1e-6, positive_only=True, k=100,
            use_normalization=False, normalization_type='l2'):
        self.__n_users = training_set.shape[0]
        self.__n_items = training_set.shape[1]
        if self.__use_augmentation:
            augmenter = self.__augmenter_cbf(original_urm=training_set,
                                             n_users=self.__n_users,
                                             n_items=self.__n_items,
                                             n_ones=self.__n_ones,
                                             n_scores=self.__n_scores)
            self.__new_urm = augmenter.get_new_urm()
        self.__training_set = training_set
        if self.__use_augmentation:
            self.__learner = MultiThreadSLIM_ElasticNet(URM_train=self.__new_urm)
        else:
            self.__learner = MultiThreadSLIM_ElasticNet(URM_train=self.__training_set)
        self.__similarity = self.__learner.fit(l1_ratio=l1_regularization,
                                               positive_only=positive_only,
                                               topK=k,
                                               l2_regularization=l2_regularization)
        if use_normalization:
            self.__similarity = normalize(self.__similarity, norm=normalization_type, axis=1)
        print("the training of the slim elastic net is terminated!")

    def recommend(self, userId, at=10, exclude_seen=True):
        user_profile = self.__training_set[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__training_set.indptr[user_id]
        end_pos = self.__training_set.indptr[user_id + 1]
        user_profile = self.__training_set.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def get_similarity_matrix(self):
        return self.__similarity

    def get_scores(self, userId):
        user_profile = self.__training_set[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        return scores

    def get_new_urm(self, ratings_per_user):
        rows = []
        for user in range(0, self.__n_users):
            recommended_items = self.recommend(user, at=ratings_per_user)
            scores = self.get_scores(userId=user)
            row = np.zeros(shape=(self.__n_items,))
            for item in recommended_items:
                row[item] = scores[item]
            rows.append(sparse.csr_matrix(sparse.coo_matrix(row)))
        new_urm = sparse.vstack(rows, format='csr')
        print("the new URM matrix is ready. Shape: ", new_urm.shape)
        return new_urm

    class __augmenter_cbf:
        def __init__(self, original_urm, n_users, n_items, n_ones, n_scores):
            self.__augmenter = cbf_recommender()
            self.__augmenter.fit(training_set=original_urm,
                                 k=90,
                                 shrink=60,
                                 similarity='jaccard')
            rows = []
            index = 0
            for user in range(0, n_users):
                recommended_items = self.__augmenter.recommend(user, at=n_ones + n_scores)
                scores = self.__augmenter.get_scores(userId=user)
                maximum = scores.max()
                row = np.zeros(shape=(n_items,))
                for item in recommended_items:
                    if index <= n_ones - 1:
                        row[item] = 1
                        index += 1
                    else:
                        row[item] = scores[item] / maximum
                row = np.array(row)
                row = np.nan_to_num(row)
                rows.append(sparse.csr_matrix(sparse.coo_matrix(row)))
            self.__new_urm = sparse.vstack(rows, format='csr') + original_urm
            greater_than_one = self.__new_urm > 1
            self.__new_urm[greater_than_one] = 1
            print("the new URM matrix is ready. Shape: ", self.__new_urm.shape)

        def get_new_urm(self):
            return self.__new_urm
