from Recommender import Recommender
import numpy.random as random


class random_recommender(Recommender):

    def __init__(self):
        self.__umItems = 0

    def fit(self, training_set):
        self.__numItems = training_set.shape[1]

    def recommend(self, userId, at):
        recommended_items = random.choice(self.__numItems, at)
        return recommended_items
