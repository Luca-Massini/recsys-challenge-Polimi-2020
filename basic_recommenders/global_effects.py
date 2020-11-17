import numpy as np
from Recommender import Recommender


class GlobalEffectsRecommender(Recommender):

    def __init__(self):
        self.__URM_train = None
        self.__bestRatedItems = None

    def fit(self, training_set):
        self.__URM_train = training_set
        globalAverage = np.mean(training_set.data)
        URM_train_unbiased = training_set.copy()
        URM_train_unbiased.data -= globalAverage

        # User Bias
        user_mean_rating = URM_train_unbiased.mean(axis=1)
        user_mean_rating = np.array(user_mean_rating).squeeze()

        # In order to apply the user bias we have to change the rating value
        # in the URM_train_unbiased inner data structures
        # If we were to write:
        # URM_train_unbiased[user_id].data -= user_mean_rating[user_id]
        # we would change the value of a new matrix with no effect on the original data structure
        for user_id in range(len(user_mean_rating)):
            start_position = URM_train_unbiased.indptr[user_id]
            end_position = URM_train_unbiased.indptr[user_id + 1]

            URM_train_unbiased.data[start_position:end_position] -= user_mean_rating[user_id]

        # Item Bias
        item_mean_rating = URM_train_unbiased.mean(axis=0)
        item_mean_rating = np.array(item_mean_rating).squeeze()
        self.__bestRatedItems = np.argsort(item_mean_rating)
        self.__bestRatedItems = np.flip(self.__bestRatedItems, axis=0)

    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.__bestRatedItems, self.__URM_train[user_id].indices,
                                        assume_unique=True, invert=True)
            unseen_items = self.__bestRatedItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.__bestRatedItems[0:at]
        return recommended_items
