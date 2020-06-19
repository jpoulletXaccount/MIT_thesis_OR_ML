from src.data_set_creation.clustering_process import clustering_name


class Config(object):
    """
    Object which represents the configuration for the routing phase
    """

    def __init__(self,cap):
        self.capacity_cst =cap
        self._origin_filename = None
        self._clustering_method = None

    def dump_to_file_capacity(self,file):
        """
        Dump in the correct format the capacity constraint
        :param file: the file
        :return:
        """
        text_capacity = "CAPACITY : " + str(self.capacity_cst)
        file.write(text_capacity + "\n")

    @property
    def file(self):
        assert not self._origin_filename is None
        return self._origin_filename

    @property
    def cluster_method(self):
        assert not self._clustering_method is None
        return self._clustering_method

    @cluster_method.setter
    def cluster_method(self,value_cluster):
        assert value_cluster in [e.value for e in clustering_name.ClusteringMethod]
        self._clustering_method = clustering_name.ClusteringMethod.from_string(value_cluster)




