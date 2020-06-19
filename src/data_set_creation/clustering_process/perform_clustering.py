from src.data_set_creation.clustering_process import clustering_name, cluster
from sklearn.cluster import KMeans


import random
import numpy as np

class StopClustering(object):

    def __init__(self):
        """
        Cluster orders together based on a distance measure between orders
        """
        self.method_mapping = {
            clustering_name.ClusteringMethod.RANDOM: self.perform_random_clustering,
            clustering_name.ClusteringMethod.KMEAN: self.perform_kmean_clustering,
            clustering_name.ClusteringMethod.NAIVE: self.perform_naive_clustering
        }
        self.manager_cluster = None


    def perform_clustering(self,manager_stop,config,*args):
        """
        perform clustering according to the method in the config
        :param manager_stop: a manger stop containing all stops
        :param config: the config of the problem we want to treat
        :return: a list of clusters
        """
        return self.method_mapping[config.cluster_method](manager_stop,config,*args)


    def perform_random_clustering(self,manager_stop,config):
        """
        Run a full random clustering
        :param manager_stop: an object maanger stop
        :param config: a config
        :return: a list of cluster
        """
        max_cap = config.capacity_cst
        first_cluster_list = self.random_clustering(manager_stop,0.65 * max_cap)
        second_cluster_list = self.random_clustering(manager_stop, 1.15 * max_cap)
        for i,cluster_second in enumerate(second_cluster_list):
            cluster_second.guid = "cluster_" + str(i + len(first_cluster_list))

        first_cluster_list.extend(second_cluster_list)
        return first_cluster_list


    @staticmethod
    def random_clustering(manager_stop,cap):
        """
        Perform a random clustering.
        More precisely we pick customer randomly among those not yet labeled, until we reach the cap
        :param manager_stop: a manager stop containing all stops
        :param cap: the maximal cap
        :return: a list of cluster
        """
        list_stop_id = list(manager_stop.keys())
        random.shuffle(list_stop_id)

        current_demand =0
        current_list_stop = []
        list_cluster = []
        for stop_id in list_stop_id:
            demand = manager_stop[stop_id].demand
            if current_demand + demand <= cap:
                current_demand += demand
                current_list_stop.append(manager_stop[stop_id])
            else:
                # Need to build a cluster with the stops
                cluster_id = "cluster_" + str(len(list_cluster))
                list_cluster.append(cluster.Cluster(cluster_id, current_list_stop))

                # put the stop in a new cluster
                current_demand = demand
                current_list_stop = [manager_stop[stop_id]]

        assert len(current_list_stop) >0
        # We create the last cluster
        cluster_id = "cluster_" + str(len(list_cluster))
        list_cluster.append(cluster.Cluster(cluster_id, current_list_stop))

        return list_cluster


    def perform_kmean_clustering(self,manager_stop,config):
        """
        Create clusters based on Kmean algorithm, whose number of clusters is determined by the cap of the config
        :param manager_stop: a manager stop of all stops
        :param config: the config corresponding to the problem
        :return: a list of clusters
        """
        max_cap = config.capacity_cst
        total_demand = manager_stop.demand

        first_cluster_list = self.kmean_clustering(manager_stop,int(total_demand/(0.85 * max_cap)))
        second_cluster_list = self.kmean_clustering(manager_stop, int(total_demand/(1.55 * max_cap)))
        third_cluster_list = self.kmean_clustering(manager_stop, int(total_demand/(1.35 * max_cap)))
        for i,cluster_second in enumerate(second_cluster_list):
            cluster_second.guid = "cluster_" + str(i + len(first_cluster_list))

        first_cluster_list.extend(second_cluster_list)

        for i,cluster_second in enumerate(third_cluster_list):
            cluster_second.guid = "cluster_" + str(i + len(first_cluster_list))

        first_cluster_list.extend(third_cluster_list)
        return first_cluster_list


    def perform_naive_clustering(self,manager_stop,config,perc):
        """
        Only performs one round of Kmean, for the perc precised
        :param manager_stop:  a manager stop of all stops
        :param config:the config corresponding to the problem
        :param perc: influence the number of cluster, based on the percentage of capacity
        :return: a list of clusters
        """
        max_cap = config.capacity_cst
        total_demand = manager_stop.demand

        first_cluster_list = self.kmean_clustering(manager_stop,int(total_demand/(perc * max_cap)))

        return first_cluster_list


    @staticmethod
    def kmean_clustering(manager_stop,number_cluster):
        """
        Perform a Kmean algorithm on the given stops with a given number of clusters
        :param manager_stop: a manger_stop
        :param number_cluster: the number of clusters
        :return: a list of clusters
        """
        matrix_array = np.array([[stop.x,stop.y] for stop in manager_stop.values()])

        clusters_list_stop = KMeans(n_clusters = number_cluster).fit_predict(X=matrix_array)
        dict_cluster_stops = dict()
        for i,stop_id in enumerate(list(manager_stop.keys())):
            cluster_id = clusters_list_stop[i]

            if not cluster_id in dict_cluster_stops.keys():
                dict_cluster_stops[cluster_id] = []
            dict_cluster_stops[cluster_id].append(manager_stop[stop_id])

        # Create the cluster
        list_final_cluster = []
        for cluster_id in dict_cluster_stops:
            guid = "cluster_" + str(cluster_id)
            list_final_cluster.append(cluster.Cluster(guid, dict_cluster_stops[cluster_id]))

        return list_final_cluster



