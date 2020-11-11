import numpy as np
import scipy.sparse as sparse
from utils.data_manager.data_manager import data_manager
from sklearn.model_selection import train_test_split


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
        train_ = sparse.csr_matrix(
            (self.__URM.data[train_mask], (self.__URM.nonzero()[0][train_mask], self.__URM.indices[train_mask])))
        test_ = sparse.csr_matrix(
            (self.__URM.data[test_mask], (self.__URM.nonzero()[0][test_mask], self.__URM.indices[test_mask])))
        return train_, test_

    @staticmethod
    def __assert_train_test_val_percentage(percentage_of_training_data, percentage_of_validation_data,
                                           percentage_of_testing_data=None):
        assert (percentage_of_validation_data > 0 and percentage_of_training_data > 0
                and (percentage_of_testing_data is None or percentage_of_testing_data > 0))
        if percentage_of_testing_data is not None:
            assert (percentage_of_validation_data + percentage_of_training_data + percentage_of_testing_data == 1)
        else:
            assert (percentage_of_training_data + percentage_of_validation_data < 1)

    def random_hold_out_ratings_validation(self, percentage_of_training_data, percentage_of_validation_data,
                                           percentage_of_testing_data=None):
        self.__assert_train_test_val_percentage(percentage_of_training_data, percentage_of_validation_data,
                                                percentage_of_testing_data)
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

    @staticmethod
    def __unzip_couples(couples1, couples2):
        users_1 = [elem[0] for elem in couples1]
        users_2 = [elem[0] for elem in couples2]
        items_1 = [elem[1] for elem in couples1]
        items_2 = [elem[1] for elem in couples2]
        return users_1, users_2, items_1, items_2

    def get_train_test(self, percentage_of_training_data=0.8):
        assert (0 < percentage_of_training_data < 1)
        users, items = self.__data_importer.get_non_zero_urm_coordinates()
        n_users, n_items = np.max(users) + 1, np.max(items) + 1
        couples = list(zip(users, items))
        couples = [list(elem) for elem in couples]
        data = self.__URM.data
        couples_train, couples_test, data_train, data_test = train_test_split(
            couples, data,
            test_size=1 - percentage_of_training_data,
            shuffle=True,
            random_state=1234
        )
        users_training, users_testing, items_training, items_testing = self.__unzip_couples(couples_train, couples_test)
        urm_training = sparse.coo_matrix((data_train, (users_training, items_training)), shape=(n_users, n_items))
        urm_training = sparse.csr_matrix(urm_training)
        urm_testing = sparse.coo_matrix((data_test, (users_testing, items_testing)), shape=(n_users, n_items))
        urm_testing = sparse.csr_matrix(urm_testing)
        return urm_training, urm_testing

    def get_train_evaluation_test(self, percentage_of_training_data, percentage_of_validation_data,
                                  percentage_of_testing_data=None):
        self.__assert_train_test_val_percentage(percentage_of_training_data, percentage_of_validation_data,
                                                percentage_of_testing_data)
        tmp = percentage_of_training_data
        percentage_of_training_data = percentage_of_training_data + percentage_of_validation_data
        urm_training, urm_testing = self.get_train_test(percentage_of_training_data)
        percentage_of_validation_data = percentage_of_validation_data / tmp
        users, items = self.__data_importer.get_non_zero_urm_coordinates()
        n_users, n_items = np.max(users) + 1, np.max(items) + 1
        couples = list(zip(users, items))
        couples = [list(elem) for elem in couples]
        data = urm_training.data
        couples_train, couples_validation, data_train, data_validation = train_test_split(
            couples, data,
            test_size=percentage_of_validation_data,
            shuffle=True,
            random_state=1234
        )
        users_training, users_validation, items_training, items_validation = self.__unzip_couples(couples_train,
                                                                                                  couples_validation)
        urm_training = sparse.coo_matrix((data_train, (users_training, items_training)), shape=(n_users, n_items))
        urm_training = sparse.csr_matrix(urm_training)
        urm_validation = sparse.coo_matrix((data_validation, (users_validation, items_validation)),
                                           shape=(n_users, n_items))
        urm_validation = sparse.csr_matrix(urm_validation)
        return urm_training, urm_validation, urm_testing

    def get_all_data(self):
        return self.__URM
