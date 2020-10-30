import numpy as np
import scipy.sparse as sparse
import pandas as pd
import scipy as scp


class data_manager:
    def __init__(self):
        self.data_in_files = []
        # The names of the files to be downloaded. Until now they're empty strings since the competition is not opened
        # yet. Both file_names array and names array will be static. The same holds for the indexes
        self.__file_names = ["", "", ""]
        self.__icm_files = [1, 2, 3]
        self.__ucm_files = [0, 4, 5]
        self.__names = ["", "", ""]
        self.__data_in_files = []
        self.__download_files_offline()
        self.__ICM = self.__compute_icm()
        self.__UCM = self.__compute_ucm()

    def get_icm(self):
        return self.__ICM

    def get_ucm(self):
        return self.__UCM

    def __download_files_offline(self):
        for file in self.__file_names:
            self.data_in_files.append(pd.read_csv(file).to_numpy())

    def __compute_icm(self):
        return self.__compute_matrix()

    def __compute_ucm(self):
        return self.__compute_matrix(ICM=False)

    def __compute_matrix(self, ICM=True):
        if ICM:
            matrices = [self.__data_in_files[index] for index in self.__icm_files]
        else:
            matrices = [self.__data_in_files[index] for index in self.__ucm_files]
        length = len(matrices)
        rows_per_matrix = [matrix[:, 0] for matrix in matrices]
        columns_per_matrix = [matrix[:, 1] for matrix in matrices]
        data_per_matrix = [matrix[:, 2] for matrix in matrices]
        matrices = [sparse.coo_matrix(data_per_matrix[index], (rows_per_matrix[index], columns_per_matrix[index])) for index in range(0,length)]
        matrices = [sparse.csr_matrix(matrix) for matrix in matrices]
        result = sparse.hstack(matrices)
        return result

