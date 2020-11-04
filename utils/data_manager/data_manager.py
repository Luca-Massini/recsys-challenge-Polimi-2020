import numpy as np
import scipy.sparse as sparse
import pandas as pd


# This class builds all the matrices and imports also all the files from the data folder in the right way.
class data_manager:
    __instance = None

    class __data_manager_instance:
        def __init__(self):
            # The names of the files to be downloaded. Until now they're empty strings since the competition is not opened
            # yet. Both file_names array and names array will be static. The same holds for the indexes
            self.__file_names = ["data_ICM_asset.csv", "data_ICM_price.csv", "data_ICM_sub_class.csv",
                                 "data_UCM_age.csv",
                                 "data_UCM_region.csv", "data_train.csv"]
            self.__icm_files = [0, 1, 2]
            self.__ucm_files = [3, 4]
            self.__urm_files = [5]
            self.__data_in_files = []
            self.__download_files_offline()
            self.__URM, self.__ICM, self.__UCM = self.__compute_matrices()

        # This method returns the URM matrix in csr matrix form
        def get_urm(self):
            return self.__URM

        # This method returns the UCM matrix in csr matrix form
        def get_ucm(self):
            return self.__UCM

        # This method returns the ICM matrix in csr matrix form
        def get_icm(self):
            return self.__ICM

        # This method imports the data from the csv files contained in the csv data folder
        def __download_files_offline(self):
            for file in self.__file_names:
                file = "../../data/" + file
                self.__data_in_files.append(pd.read_csv(file).to_numpy())

        # This method casts any array into integer numpy array
        @staticmethod
        def __int_cast(array):
            return np.array(list(map(int, array)))

        # This method takes as input some matrices with the same number of columns but possibly a different number of rows. It
        # resize all the matrices in such a way that in the end they will have the same number of rows
        @staticmethod
        def __equalize_rows(matrices):
            row = np.max([matrix.shape[0] for matrix in matrices])
            for matrix in matrices:
                matrix.resize((row, matrix.shape[1]))

        # It computes and builds the URM, ICM and UCM matrix (in sparse.csr_matrix format) and it gives them as result
        def __compute_matrices(self):
            files = [self.__urm_files, self.__ucm_files, self.__icm_files]
            matrices = [[self.__data_in_files[i] for i in index] for index in files]
            urm_matrices, ucm_matrices, icm_matrices = matrices[0], matrices[1], matrices[2]
            urm_coo_matrices = [
                sparse.coo_matrix((matrix[:, 2], (self.__int_cast(matrix[:, 0]), self.__int_cast(matrix[:, 1])))) for
                matrix in urm_matrices]
            icm_coo_matrices = [
                sparse.coo_matrix((matrix[:, 2], (self.__int_cast(matrix[:, 0]), self.__int_cast(matrix[:, 1])))) for
                matrix in icm_matrices]
            ucm_coo_matrices = [
                sparse.coo_matrix((matrix[:, 2], (self.__int_cast(matrix[:, 0]), self.__int_cast(matrix[:, 1])))) for
                matrix in ucm_matrices]
            self.__equalize_rows(urm_coo_matrices)
            self.__equalize_rows(icm_coo_matrices)
            self.__equalize_rows(ucm_coo_matrices)
            URM = sparse.hstack(urm_coo_matrices, format='csr')
            ICM = sparse.hstack(icm_coo_matrices, format='csr')
            UCM = sparse.hstack(ucm_coo_matrices, format='csr')
            return URM, ICM, UCM

    def __new__(cls):
        if not data_manager.__instance:
            data_manager.__instance = data_manager.__data_manager_instance()
        return data_manager.__instance

    def get_urm(self):
        return self.__instance.get_urm()

    def get_ucm(self):
        return self.__instance.get_ucm()

    def get_icm(self):
        return self.__instance.get_icm()
