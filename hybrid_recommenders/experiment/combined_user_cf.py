from CF.cf_recommender_user import cf_recommender_user
from Recommender import Recommender
from hybrid_recommenders.experiment.URM_item_cbf_augmenter import URM_item_cbf_augmenter
import numpy as np


class combined_user_cf(Recommender):
    def __init__(self, k_augmenter=90, similarity_augmenter='jaccard', shrink_augmenter=60):
        self.__user_cf_learner_1 = cf_recommender_user()
        self.__k_aug = k_augmenter
        self.__similarity_aug = similarity_augmenter
        self.__shrink_aug = shrink_augmenter
        self.__enriched_urm = None
        self.__original_urm = None
        self.__similarity_matrix = None

    def fit(self, training_set, k=100, shrink=10, similarity='cosine', rating_per_user=5):
        self.__original_urm = training_set
        self.__enriched_urm = URM_item_cbf_augmenter(training_set=training_set.copy(),
                                                     k=self.__k_aug,
                                                     similarity=self.__similarity_aug,
                                                     shrink=self.__shrink_aug,
                                                     rating_per_users=rating_per_user).get_new_urm()
        print("the new urm is ready")
        self.__user_cf_learner_1.fit(training_set=self.__enriched_urm.copy(),
                                     k=k,
                                     shrink=shrink,
                                     similarity=similarity)
        self.__similarity_matrix = self.__user_cf_learner_1.get_similarity_matrix()

    def recommend(self, userId, at=None, exclude_seen=True):
        scores = self.__similarity_matrix[userId, :].dot(self.__original_urm).toarray().ravel()
        scores = self.__filter_seen(userId, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def get_similarity_matrix(self):
        return self.__similarity_matrix

    def __filter_seen(self, user_id, scores):
        start_pos = self.__original_urm.indptr[user_id]
        end_pos = self.__original_urm.indptr[user_id + 1]
        user_profile = self.__original_urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def change_model_parameters(self, new_k, new_shrink, new_similarity):
        self.__user_cf_learner_1 = cf_recommender_user()
        self.__user_cf_learner_1.fit(training_set=self.__enriched_urm,
                                     k=new_k,
                                     shrink=new_shrink,
                                     similarity=new_similarity)
        self.__similarity_matrix = self.__user_cf_learner_1.get_similarity_matrix()

    def get_enriched_urm(self):
        return self.__enriched_urm
