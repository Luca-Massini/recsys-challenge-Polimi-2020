import numpy as np
import pandas as pd
import os
import csv
from basic_recommenders.TopPopRecommender import TopPopRecommender

from Recommender import Recommender
from utils.data_manager.data_manager import data_manager


class submission_helper:
    def __init__(self, name_of_the_file, recommender: Recommender, at=10):
        self.__recommender = recommender
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../data/" + "data_target_users_test.csv")
        self.__users = pd.read_csv(filename).to_numpy().ravel()
        self.__recommended_items_per_user = [self.__recommender.recommend(user, at) for user in self.__users]
        self.__name_of_the_file = name_of_the_file

    def build_submission(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../submissions/" + self.__name_of_the_file + ".csv")
        num_users = self.__users.shape[0]
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "item_list"])
            for index in range(0, num_users):
                recommendations = self.__recommended_items_per_user[index]
                recommendations = recommendations.astype(np.str)
                recommendations = [', '.join(recommendations).replace(',', '')]
                user = self.__users[index]
                row = np.concatenate([[user], recommendations])
                writer.writerow(row)
