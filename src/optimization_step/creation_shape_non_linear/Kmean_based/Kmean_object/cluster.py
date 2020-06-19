
from src.data_set_creation.stops.manager import stops_manager_cvrptw

class Cluster(stops_manager_cvrptw.StopsManagerCVRPTW):
    """
    Instancie a cluster class
    """
    def __init__(self,depot,guid):
        super(Cluster,self).__init__(depot)
        self.guid = guid
        self.prediction = None
        self.expected_prediction = None

    def set_prediction(self,predict):
        """
        :param predict: the class label
        :return:
        """
        self.prediction = predict

    def set_expected_prediction(self, expected):
        """
        :param expected: the expectation of the class label
        :return:
        """
        self.expected_prediction = expected

    def set_expected_prediction_via_proba(self, dict_proba):
        """
        :param dict_proba: the proba for each class
        :return:
        """
        final_class = 0
        for label,proba in dict_proba.items():
            final_class += label * proba
        self.expected_prediction = final_class


    def add_stop(self,stop):
        """
        Add a stop to the clusters
        :param stop: the stop to be added
        :return:
        """
        assert not stop.guid in self.keys(), stop.guid
        self[stop.guid] = stop


    def remove_stop(self,stop):
        """
        :param stop: the stop to be remove from the clusters
        :return:
        """
        assert stop.guid in self.keys(), stop.guid + str(self.keys())
        del self[stop.guid]




