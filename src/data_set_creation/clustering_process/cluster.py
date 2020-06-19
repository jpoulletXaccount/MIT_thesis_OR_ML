import numpy as np


class Cluster(object):
    def __init__(self, guid, list_stops):
        """
        A cluster is a group of stops

        CLUSTERS CANNOT BE CHANGED ONCE THEY ARE INITIALIZED
        :param guid: id of the clusters
        :param list_stops: list of stops part of the cluster
        """
        self.guid = guid
        self.list_stops = list_stops

        # to be initialized later
        self._stops_centroid = None
        self._stops_centroid_std = None

        # Only useful for coordinator
        self.depot = None
        self.matrix_depot_dist = None
        self.matrix_stops_dist = None

    def __len__(self):
        """
        How many stops in the cluster
        :return: number of stops
        """
        return len(self.list_stops)

    @property
    def total_demand(self):
        """
        Total demand to be delivered as part of this cluster
        :return:
        """
        return sum(stop.demand for stop in self.list_stops)


    @property
    def stop_centroid(self):
        """
        Calculates centroid of stops in relative utm coords
        :return: (x, y)
        """
        if self._stops_centroid is None:
            self._stops_centroid = np.mean([stop.xy for stop in self.list_stops])

        return self._stops_centroid

    @property
    def cust_centroid_std(self):
        """
        Calculates standard deviation of the stops in relative utm coords
        :return: (x, y)
        """
        if self._stops_centroid_std is None:
            self._stops_centroid_std = np.std([stop.xy for stop in self.list_stops])

        return self._stops_centroid_std

