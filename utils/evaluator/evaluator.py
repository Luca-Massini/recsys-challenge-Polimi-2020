import numpy as np

from Recommender import Recommender
from utils.data_manager.splitter import splitter
from utils.evaluator.evaluator_component import evaluator_component


class evaluator:

    def __init__(self, recommender_object: Recommender):
        self.__recommender = recommender_object

    def repeated_hold_out(self, iterations=20, percentage_of_training_data=0.8, at=10):
        precision_array = []
        recall_array = []
        map_array = []
        for experiment in range(0, iterations):
            print(experiment + 1)
            train, test = splitter().random_hold_out_ratings_train_test(
                percentage_of_training_data=percentage_of_training_data)
            self.__recommender.fit(train)
            current_precision, current_recall, current_map = evaluator_component(trained_recommender=self.__recommender,
                                                                                 testing_matrix=test).evaluate(at=at)
            precision_array.append(current_precision)
            recall_array.append(current_recall)
            map_array.append(current_map)
        return np.mean(precision_array), np.mean(recall_array), np.mean(map_array)

    def evaluate(self, percentage_of_training_data=0.8, at=10):
        train, test = splitter().get_train_test(percentage_of_training_data=percentage_of_training_data)
        self.__recommender.fit(training_set=train)
        precision, recall, map_ = evaluator_component(trained_recommender=self.__recommender,
                                                      testing_matrix=test).evaluate(at)
        return precision, recall, map_

    @staticmethod
    def evaluate_already_trained(recommender: Recommender, percentage_of_training_data=0.6, percentage_of_validation_data=0.2, at=10):
        train, validation, test = splitter().get_train_evaluation_test(percentage_of_training_data=percentage_of_training_data,
                                                                       percentage_of_validation_data=percentage_of_validation_data)
        precision, recall, map_ = evaluator_component(trained_recommender=recommender,
                                                      testing_matrix=validation).evaluate(at)
        return precision, recall, map_

    def compute_validation_score(self, validation_set, training_set, at=10):
        self.__recommender.fit(training_set=training_set)
        precision, recall, map_ = evaluator_component(trained_recommender=self.__recommender,
                                                      testing_matrix=validation_set).evaluate(at=at)
        return precision, recall, map_
