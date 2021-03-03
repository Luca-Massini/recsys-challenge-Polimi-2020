from Recommender import Recommender
from hybrid_recommenders.experiment.URM_item_cbf_augmenter import URM_item_cbf_augmenter
from utils.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
import numpy as np
import scipy.sparse as sparse

from utils.data_manager.data_manager import data_manager

SEED = 1234


class pure_SVD_recommender(Recommender):
    def __init__(self, use_augmented_urm=False):
        self.__urm_training = None
        self.__learner = None
        self.__similarity = None
        self.__icm = data_manager().get_icm()
        self.__n_items = None
        self.__n_users = None
        self.__original_urm = None
        self.__use_augmentation = use_augmented_urm
        self.__augmenter = None
        self.__use_augmentation = use_augmented_urm
        self.__augmented_urm = None

    def fit(self, training_set, num_factors=40, k=100, concatenate_icm=False):
        self.__augmenter = URM_item_cbf_augmenter(training_set=training_set)
        if self.__use_augmentation:
            self.__augmented_urm = self.__augmenter.get_new_urm()
        self.__original_urm = training_set
        self.__n_users = training_set.shape[0]
        self.__n_items = training_set.shape[1]
        self.__urm_training = training_set
        if not self.__use_augmentation:
            new_urm = self.__urm_training
        else:
            new_urm = self.__augmented_urm
        if concatenate_icm:
            new_urm = sparse.vstack([self.__icm.T, new_urm])
        self.__learner = PureSVDItemRecommender(URM_train=new_urm.copy())
        self.__similarity = self.__learner.fit(num_factors=num_factors, topK=k, random_seed=SEED)

    def recommend(self, userId, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        if exclude_seen:
            scores = self.__filter_seen(userId, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def get_scores(self, userId):
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        return scores

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]
        user_profile = self.__urm_training.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def get_similarity_matrix(self):
        return self.__similarity

    def get_new_urm(self, ratings_per_user):
        print("I start to build the new_urm")
        rows = []
        for user in range(0, self.__n_users):
            recommended_items = self.recommend(user, at=ratings_per_user, exclude_seen=False)
            scores = self.get_scores(userId=user)
            row = np.zeros(shape=(self.__n_items,))
            for item in recommended_items:
                row[item] = scores[item]
            rows.append(sparse.csr_matrix(sparse.coo_matrix(row)))
        new_urm = sparse.vstack(rows, format='csr')
        print("the new URM matrix is ready. Shape: ", new_urm.shape)
        return new_urm
