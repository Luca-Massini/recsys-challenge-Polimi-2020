from CBF.cbf_recommender import cbf_recommender
from Recommender import Recommender
import numpy as np


class cbf_recommender_cold_users(cbf_recommender):
    def __init__(self, non_personalized_recommender: Recommender, icm=None, min_ratings=-1):
        super(cbf_recommender_cold_users, self).__init__(icm=icm)
        self.__non_personalized_recommender = non_personalized_recommender
        self.__cold_users_rows = []
        self.__number_of_ratings_per_users = None
        self.__threshold = min_ratings

    def fit(self, training_set, **kwargs):
        super(cbf_recommender_cold_users, self).fit(training_set=training_set)
        self.__non_personalized_recommender.fit(training_set=training_set)
        self.__cold_users_rows = np.diff(training_set.indptr) == 0
        self.__cold_users_rows = np.array(
            [i for i in range(0, len(self.__cold_users_rows)) if self.__cold_users_rows[i]])
        n_users = training_set.shape[0]
        self.__number_of_ratings_per_users = [training_set[user].indptr for user in range(0, n_users)]
        self.__number_of_ratings_per_users = [len(array) for array in self.__number_of_ratings_per_users]

    def recommend(self, userId, at=10, exclude_seen=True):
        if userId in self.__cold_users_rows:
            recommendations = self.__non_personalized_recommender.recommend(userId=userId, at=at)
        else:
            if self.__number_of_ratings_per_users[userId] > self.__threshold:
                recommendations = super(cbf_recommender_cold_users, self).recommend(userId=userId, at=at)
            else:
                recommendations = self.__non_personalized_recommender.recommend(userId=userId, at=at)
        return recommendations
