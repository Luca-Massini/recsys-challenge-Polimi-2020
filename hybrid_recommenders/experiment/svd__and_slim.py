from GRAPH_BASED.graph_based_recommender_alpha import graph_based_recommender_alpha
from Recommender import Recommender
from SLIM_ElasticNet.slim_elastic_net import slim_elastic_net
from SVD.pure_SVD_recommender import pure_SVD_recommender
from hybrid_recommenders.experiment.URM_item_cbf_augmenter import URM_item_cbf_augmenter
import numpy as np


class svd_and_slim(Recommender):
    def __init__(self, weight):
        self.__urm_training = None
        self.__augmented_matrix = None
        self.__graph_similarity = None
        self.__slim_learner = None
        self.__graph_learner = None
        self.__slim_similarity_matrix = None
        self.__graph_similarity = None
        self.__weight = weight

    def fit(self, training_set, l1_reg=5e-3, l2_reg=0.005, k_slim=300):
        self.__urm_training = training_set
        self.__augmented_matrix = URM_item_cbf_augmenter(training_set=training_set,
                                                         rating_per_users=3).get_new_urm()
        self.__slim_learner = slim_elastic_net()
        self.__slim_learner.fit(training_set=self.__augmented_matrix.copy(),
                                l1_regularization=l1_reg,
                                l2_regularization=l2_reg,
                                k=k_slim)
        self.__slim_similarity_matrix = self.__slim_learner.get_similarity_matrix()
        self.__graph_learner = graph_based_recommender_alpha()
        self.__graph_learner.fit(training_set=self.__augmented_matrix,
                                 alpha=0.5,
                                 k=200)
        self.__graph_similarity = self.__graph_learner.get_similarity_matrix()

    def recommend(self, userId, at):
        user_profile = self.__urm_training[userId]
        scores_slim = user_profile.dot(self.__slim_similarity_matrix).toarray().ravel()
        scores_slim /= scores_slim.max()
        scores_svd = user_profile.dot(self.__graph_similarity).toarray().ravel()
        scores_svd /= scores_svd.max()
        scores = scores_svd * self.__weight + scores_slim * (1 - self.__weight)
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]
        user_profile = self.__urm_training.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def change_weight(self, new_weight):
        self.__weight = new_weight
