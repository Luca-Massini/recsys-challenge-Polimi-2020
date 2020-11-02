import numpy as np
import scipy.sparse as sparse
from utils.data_manager.data_manager import data_manager

# so every time I will split the data, it will will be divided in the same way every single time
np.random.RandomState = 1234


class splitter:
    def __init__(self):
        self.__data_importer = data_manager()
        # matrix not splitted at all
        self.__URM = self.__data_importer.get_urm()

    def random_hold_out_ratings_train_test(self, percentage_of_training_data):
        assert (0 < percentage_of_training_data < 1)
        num_interactions = self.__URM.nnz
        train_mask = np.random.choice([True, False],
                                      num_interactions,
                                      p=[percentage_of_training_data, 1 - percentage_of_training_data])
        test_mask = np.logical_not(train_mask)
        URM_train = sparse.csr_matrix(
            (self.__URM.data[train_mask], (self.__URM.nonzero()[0][train_mask], self.__URM.indices[train_mask])))
        URM_test = sparse.csr_matrix(
            (self.__URM.data[test_mask], (self.__URM.nonzero()[0][test_mask], self.__URM.indices[test_mask])))
        return URM_train, URM_test

    def random_hold_out_ratings_validation(self, percentage_of_training_data, percentage_of_validation_data, percentage_of_testing_data=None):
        assert (percentage_of_validation_data > 0 and percentage_of_training_data > 0 and (
                    percentage_of_testing_data is None or percentage_of_testing_data > 0))
        if percentage_of_testing_data is not None:
            assert (percentage_of_validation_data + percentage_of_training_data + percentage_of_testing_data == 1)
        else:
            assert (percentage_of_training_data + percentage_of_validation_data < 1)
        tmp = percentage_of_training_data
        percentage_of_training_data = percentage_of_training_data + percentage_of_validation_data
        train_, test_ = self.random_hold_out_ratings_train_test(percentage_of_training_data)
        percentage_of_validation_data = percentage_of_validation_data / tmp
        num_interactions = train_.nnz
        validation_mask = np.random.choice([True, False],
                                           num_interactions,
                                           p=[percentage_of_validation_data, 1 - percentage_of_validation_data])
        train_mask = np.logical_not(validation_mask)
        tmp = train_.copy()
        train_ = sparse.csr_matrix(
            (train_.data[train_mask], (train_.nonzero()[0][train_mask], train_.indices[train_mask])))
        validation_ = sparse.csr_matrix(
            (tmp.data[validation_mask], (tmp.nonzero()[0][validation_mask], tmp.indices[validation_mask])))
        return train_, validation_, test_

    def get_train_test_together(self):
        return self.__URM
