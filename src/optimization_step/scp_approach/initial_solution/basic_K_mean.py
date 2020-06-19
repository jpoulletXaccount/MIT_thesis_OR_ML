import numpy as np
from sklearn.cluster import KMeans


class basicKMeans(object):
    """
    Class which runs a basic K mean on a list of stops and return clusters
    """
    def __init__(self,manager_cluster,manager_stops):
        self.manager_stops = manager_stops
        self.manager_clusters = manager_cluster



    def run_K_mean(self,list_stops, K):
        """
        Perform the usual K mean based on stops distance
        :param list_stops: the list of stops
        :param K: the number of clusters we want
        Update self.manager
        """
        dict_Kmean_cluster = self._kmean_clustering(list_stops,K)
        tracker = 'Kmean_' + str(K)
        for list_stop_id in dict_Kmean_cluster.values():
            self.manager_clusters.create_cluster(list_stop_id, self.manager_stops,(tracker,0))



    def _kmean_clustering(self,list_stops_id,K):
        """
        Perform a Kmean algorithm on the given stops with a given number of clusters
        :param list_stops_id: the list of stops we want to cluster
        :param K: the number of clusters
        :return: a dict[cluster_id] = list of stops_id
        """
        list_stop = [self.manager_stops[stop_id] for stop_id in list_stops_id]
        matrix_array = np.array([[stop.x,stop.y] for stop in list_stop])

        clusters_list_stop = KMeans(n_clusters = K).fit_predict(X=matrix_array)
        dict_cluster_stops = dict()
        for i,stop_id in enumerate(list_stops_id):
            cluster_id = clusters_list_stop[i]

            if not cluster_id in dict_cluster_stops.keys():
                dict_cluster_stops[cluster_id] = []
            dict_cluster_stops[cluster_id].append(stop_id)

        return dict_cluster_stops
