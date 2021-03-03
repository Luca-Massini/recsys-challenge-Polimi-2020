from CBF.cbf_recommender import cbf_recommender
from CF.cf_recommender_user import cf_recommender_user
from GRAPH_BASED.graph_based_recommender_alpha import graph_based_recommender_alpha
from Recommender import Recommender
from SLIM_ElasticNet.slim_elastic_net import slim_elastic_net
from hybrid_recommenders.item_knn_cf_cbf.item_knn_cf_cbf_concatenated import item_knn_cf_cbf
import numpy as np


class hybrid_rec(Recommender):

    def __init__(self, weights, enriching_factor=5):
        self.__urm_training = None
        self.__cf_cbf_learner = None
        self.__user_cf_learner = None
        self.__graph_based_learner = None
        self.__slim_learner = None
        self.__urm_augmenter = None
        self.__cf_cbf_similarity = None
        self.__user_cf_similarity = None
        self.__graph_similarity = None
        self.__slim_similarity = None
        self.__similarity = None
        self.__weights = weights
        self.__new_urm = None
        self.__enriching_factor = enriching_factor

    def fit(self, training_set, k_cf_cbf=100, shrink_cf_cbf=30, similarity_cf_cbf='cosine',
            k_cf_user=100, shrink_cf_user=10, similarity_cf_user='cosine', k_cbf=100, similarity_cbf='cosine',
            shrink_cbf=20, k_graph=100, alpha=0.4, k_slim=100, l1_regularization=0.1, l2_regularization=0.2):
        self.__urm_training = training_set
        self.__urm_augmenter = cbf_recommender()
        self.__urm_augmenter.fit(training_set=training_set,
                                 k=k_cbf,
                                 shrink=shrink_cbf,
                                 similarity=similarity_cbf)
        new_urm = self.__urm_augmenter.get_new_urm(ratings_per_user=self.__enriching_factor)
        new_urm = self.__combine_urm_matrices(new_urm=new_urm)
        self.__new_urm = new_urm
        self.__urm_training = training_set
        self.__cf_cbf_learner = item_knn_cf_cbf()
        self.__cf_cbf_learner.fit(training_set=self.__urm_training.copy(),
                                  k=k_cf_cbf,
                                  shrink=shrink_cf_cbf,
                                  similarity=similarity_cf_cbf)
        self.__cf_cbf_similarity = self.__cf_cbf_learner.get_similarity_matrix()
        self.__cf_cbf_similarity /= self.__cf_cbf_similarity.max()
        self.__user_cf_learner = cf_recommender_user()
        self.__user_cf_learner.fit(training_set=new_urm.copy(),
                                   k=k_cf_user,
                                   shrink=shrink_cf_user,
                                   similarity=similarity_cf_user)
        self.__user_cf_similarity = self.__user_cf_learner.get_similarity_matrix()
        self.__user_cf_similarity /= self.__user_cf_similarity.max()
        self.__graph_based_learner = graph_based_recommender_alpha()
        self.__graph_based_learner.fit(training_set=self.__new_urm.copy(),
                                       k=k_graph,
                                       alpha=alpha)
        self.__graph_similarity = self.__graph_based_learner.get_similarity_matrix()
        self.__graph_similarity /= self.__graph_similarity.max()
        self.__slim_learner = slim_elastic_net()
        self.__slim_learner.fit(training_set=new_urm,
                                k=k_slim,
                                l1_regularization=l1_regularization,
                                l2_regularization=l2_regularization)
        self.__slim_similarity = self.__slim_learner.get_similarity_matrix()
        self.__slim_similarity /= self.__slim_similarity.max()
        self.__similarity = self.__slim_similarity * self.__weights[0] + self.__graph_similarity * self.__weights[
            1] + self.__cf_cbf_similarity * self.__weights[2]
        self.__similarity /= self.__similarity.max()

    def recommend(self, userId, at, normalize_scores=True):
        scores_cf_user = self.__user_cf_similarity[userId, :].dot(self.__urm_training).toarray().ravel()
        user_profile = self.__urm_training[userId]
        scores = user_profile.dot(self.__similarity).toarray().ravel()
        scores = scores * self.__weights[3] + scores_cf_user * (1 - self.__weights[3])
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def change_weights(self, new_weights):
        self.__weights = new_weights
        self.__similarity = self.__slim_similarity * self.__weights[0] + self.__graph_similarity * self.__weights[
            1] + self.__cf_cbf_similarity * self.__weights[2]
        self.__similarity /= self.__similarity.max()

    def __filter_seen(self, user_id, scores):
        start_pos = self.__urm_training.indptr[user_id]
        end_pos = self.__urm_training.indptr[user_id + 1]
        user_profile = self.__urm_training.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def __combine_urm_matrices(self, new_urm, previous_urm=None):
        if previous_urm is None:
            new_urm = new_urm + self.__urm_training
        else:
            new_urm = new_urm + previous_urm
        greater_than_one = new_urm > 1
        new_urm[greater_than_one] = 1
        return new_urm
