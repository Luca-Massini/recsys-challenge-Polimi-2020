import Recommender
import scipy.sparse as sparse
import numpy as np

from utils.data_manager.data_manager import data_manager


class evaluator_component:
    def __init__(self, trained_recommender: Recommender, is_implicit=True, threshold=2,
                 testing_matrix: sparse.csr_matrix = None):
        self.__URM_testing = testing_matrix
        self.__recommender_algorithm = trained_recommender
        self.__data_importer = data_manager()
        n_users = self.__URM_testing.shape[0]
        self.__users = [user for user in range(0, n_users)]
        if is_implicit:
            self.__relevant_items = []
            for user in self.__users:
                row_start = self.__URM_testing.indptr[user]
                row_end = self.__URM_testing.indptr[user + 1]
                non_zero_items = self.__URM_testing.indices[row_start: row_end]
                self.__relevant_items.append(non_zero_items)
        else:
            self.__relevant_items = self.__compute_relevant_explicit_ratings(threshold=threshold)

    def get_precision(self, at=10):
        recommended_items_per_user = [self.__recommender_algorithm.recommend(userId=user, at=at) for user in
                                      self.__users]
        precision_ = [self.__compute_precision(recommended_items_per_user[user], self.__relevant_items[user]) for user
                      in self.__users]
        return np.mean(precision_)

    def get_recall(self, at=10):
        recommended_items_per_user = [self.__recommender_algorithm.recommend(userId=user, at=at) for user in
                                      self.__users]
        recall_ = [self.__compute_recall(recommended_items_per_user[user], np.array(self.__relevant_items[user])) for
                   user in self.__users]
        return np.mean(recall_)

    def get_MAP(self, at=10):
        recommended_items_per_user = [self.__recommender_algorithm.recommend(userId=user, at=at) for user in
                                      self.__users]
        MAP_ = [self.__compute_MAP(recommended_items_per_user[user], np.array(self.__relevant_items[user])) for user in
                self.__users]
        return np.mean(MAP_)

    def evaluate(self, at=10):
        recommended_items_per_user = [self.__recommender_algorithm.recommend(userId=user, at=at) for user in
                                      self.__users]
        precision_ = [self.__compute_precision(recommended_items_per_user[user], self.__relevant_items[user]) for user
                      in self.__users]
        recall_ = [self.__compute_recall(recommended_items_per_user[user], np.array(self.__relevant_items[user])) for
                   user in self.__users]
        MAP_ = [self.__compute_MAP(recommended_items_per_user[user], np.array(self.__relevant_items[user])) for user in
                self.__users]
        return np.mean(precision_), np.mean(recall_), np.mean(MAP_)

    # PRIVATE METHODS:

    @staticmethod
    def __compute_precision(recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        precision_score = np.sum(is_relevant, dtype=np.float32) / (len(np.array(is_relevant)) + 1e-06)
        return precision_score

    @staticmethod
    def __compute_recall(recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        recall_score = np.sum(is_relevant, dtype=np.float32) / (relevant_items.shape[0] + 1e-6)
        return recall_score

    @staticmethod
    def __compute_MAP(recommended_items, relevant_items):
        is_relevant = np.array(np.in1d(recommended_items, relevant_items, assume_unique=True))
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / (np.min([relevant_items.shape[0], is_relevant.shape[0]]) + 1e-06)
        return map_score

    def __compute_relevant_explicit_ratings(self, threshold: int):
        relevant_items = []
        for user in self.__users:
            row_start = self.__URM_testing.indptr[user]
            row_end = self.__URM_testing.indptr[user + 1]
            row_data = self.__URM_testing.data[row_start: row_end]
            non_zero_items = self.__URM_testing.indices[row_start: row_end]
            non_zero_items_length = len(non_zero_items)
            relevant = [non_zero_items[index] for index in range(0, non_zero_items_length) if
                        row_data[index] > threshold]
            relevant_items.append(relevant)
        return relevant_items
