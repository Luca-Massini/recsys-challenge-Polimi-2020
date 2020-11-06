from Recommender import Recommender
import numpy.random as random


class random_recommender(Recommender):

    def fit(self, training_set):
        self.__numItems = training_set.shape[0]

    def recommend(self, userId, at):
        recommended_items = random.choice(self.__numItems)
        return recommended_items
