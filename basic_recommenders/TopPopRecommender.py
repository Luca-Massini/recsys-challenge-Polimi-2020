import numpy as np
from Recommender import Recommender


class TopPopRecommender(Recommender):

    def __init__(self, threshold=0, is_implicit=True):
        assert((threshold == 0 and is_implicit) or (threshold != 0 and not is_implicit))
        self.__threshold = threshold
        self.__URM_train = None
        self.__popularItems = None

    def fit(self, training_set):
        self.__URM_train = training_set

        itemPopularity = (training_set > self.__threshold).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.__popularItems = np.argsort(itemPopularity)
        self.__popularItems = np.flip(self.__popularItems, axis=0)

    def recommend(self, userId, at=5):
        unseen_items_mask = np.in1d(self.__popularItems, self.__URM_train[userId].indices,
                                    assume_unique=True, invert=True)

        unseen_items = self.__popularItems[unseen_items_mask]

        recommended_items = unseen_items[0:at]

        return recommended_items
