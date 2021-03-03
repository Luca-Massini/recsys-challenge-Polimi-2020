from CBF.cbf_recommender import cbf_recommender


class URM_item_cbf_augmenter:

    def __init__(self, training_set, k=90, shrink=60, similarity='jaccard', rating_per_users=5):
        self.__training_set = training_set
        self.__learner = cbf_recommender()
        self.__learner.fit(training_set=training_set,
                           k=k,
                           shrink=shrink, similarity=similarity)
        new_urm = self.__learner.get_new_urm(ratings_per_user=rating_per_users)
        self.__new_urm = self.__combine_matrices(new_urm=new_urm)

    def __combine_matrices(self, new_urm):
        new_urm += self.__training_set
        greater_than_one = new_urm > 1
        new_urm[greater_than_one] = 1
        return new_urm

    def get_new_urm(self):
        return self.__new_urm
