import numpy as np
import scipy.sparse as sparse
import pandas as pd


class data_manager:
    def __init__(self):
        # The names of the files to be downloaded. Until now they're empty strings since the competition is not opened
        # yet. Both file_names array and names array will be static. The same holds for the indexes
        self.__file_names = ["data_ICM_asset.csv", "data_ICM_price.csv", "data_ICM_sub_class.csv", "data_UCM_age.csv",
                             "data_UCM_region.csv"]
        self.__icm_files = [0, 1, 2]
        self.__ucm_files = [3, 4]
        self.__names_ICM = ["asset", "price", "subclass"]
        self.__names_UCM = ["age", "region"]
        self.__urm_files = []
        self.__data_in_files = []
        self.__download_files_offline()
        self.__ICM = self.__compute_icm()
        self.__UCM = self.__compute_ucm()
        self.__URM = self.__computer_urm()

    def get_items_list(self):
        return self.__item_list

    def get_icm_total(self):
        return self.__ICM.todense()

    def get_urm(self):
        if self.__URM is None:
            return None
        return self.__URM. todense()

    def get_ucm_total(self):
        if self.__UCM is None:
            return None
        return self.__UCM.todense()

    def get_icm_total_csr(self):
        if self.__ICM is None:
            return None
        return self.__ICM

    def get_ucm_total_csr(self):
        return self.__UCM

    def get_columns_names_UCM(self):
        return self.__names_UCM

    def get_columns_names_ICM(self):
        return self.__names_ICM

    def __download_files_offline(self):
        for file in self.__file_names:
            file = "../../data/" + file
            self.__data_in_files.append(pd.read_csv(file).to_numpy())
        self.__data_in_files = np.array(self.__data_in_files)

    def __compute_icm(self):
        return self.__compute_matrix()

    def __compute_ucm(self):
        return self.__compute_matrix(ICM=False)

    def __computer_urm(self):
        return self.__compute_matrix(UCM=True)

    def __set_items_list(self, icm_matrices):
        items_lists = [matrix[:, 0] for matrix in icm_matrices]
        result = set()
        for item_list in items_lists:
            result = result.union(set(item_list))
        self.__item_list = [int(elem) for elem in result]

    def __set_user_list(self, ucm_matrices):
        items_lists = [matrix[:, 0] for matrix in ucm_matrices]
        result = set()
        for user_list in items_lists:
            result = result.union(set(user_list))
        self.__item_list = [int(elem) for elem in result]

    def __compute_matrix(self, ICM=True, UCM=False):
        if UCM:
            if len(self.__urm_files) == 0:
                return None
            matrices = [self.__data_in_files[index] for index in self.__urm_files]
        else:
            if ICM:
                if len(self.__icm_files) == 0:
                    return None
                matrices = [self.__data_in_files[index] for index in self.__icm_files]
                self.__set_items_list(matrices)
            else:
                if len(self.__ucm_files) == 0:
                    return None
                matrices = [self.__data_in_files[index] for index in self.__ucm_files]
        rows = np.max([np.max(matrix[:, 0]) for matrix in matrices])+1
        columns = [np.max(matrix[:, 1]+1) for matrix in matrices]
        columns = np.sum(columns)
        result = np.zeros((int(rows), int(columns)), dtype=float)
        column = 0
        for matrix in matrices:
            limit = len(matrix)
            for index in range(0, limit):
                result[int(matrix[index][0])][int(matrix[index][1] + column)] = matrix[index][2]
            column += 1
        result = sparse.coo_matrix(result)
        result = sparse.csr_matrix(result)
        return result


if __name__ == '__main__':
    testObject = data_manager()
