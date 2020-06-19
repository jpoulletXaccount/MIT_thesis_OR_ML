
import copy
import numpy as np
from sklearn.cluster import KMeans

from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw
from src.data_set_creation.routing_model import cvrptw_routing_solver
from src.learning_step.learning_trees import find_best_tree
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset
from src.optimization_step.creation_shape_non_linear.Kmean_based.Kmean_object import manager_clusters,cluster
from src.optimization_step.creation_shape_non_linear.Kmean_based import complete_leaf,switch_leaf,safe_cluster

class create_all_cluster(object):
    """
    Create all clusters
    """
    def __init__(self,manager_stop,config, dict_cst = None, dict_label = None, dict_proba = None):
        self.manager_stop = manager_stop
        self.config = config

        # Retrieve tree etc
        if dict_cst is None or dict_proba is None or dict_label is None:
            finder = find_best_tree.FindBestTree()
            _, self.dict_cst, self.dict_label, self.dict_proba,_ = finder.find_cst_best_tree()
        else:
            self.dict_cst = dict_cst
            self.dict_label = dict_label
            self.dict_proba = dict_proba

        # To be filled
        self.manager_cluster = manager_clusters.ManagerCluster()


    def create_all_clusters(self):
        """
        Performed a K mean and then modify the cluster to get new ones
        :return: a manager cluster
        """
        total_demand = self.manager_stop.demand
        max_cap = self.config.capacity_cst
        list_number_cluster = [int(total_demand/(i * max_cap)) for i in [0.6,0.75,0.8,0.9,1,1.2,1.3,1.5]]
        # list_number_cluster = [int(total_demand/(i * max_cap)) for i in [0.85,1.35,1.55]]
        list_number_cluster = list(set(list_number_cluster))

        for k in list_number_cluster:
            list_cluster = self.kmean_clustering(number_cluster=k)
            for i in range(0,5):
                list_2 = self.kmean_clustering(number_cluster=k)
                for clu in list_2:
                    clu.guid = clu.guid.split("_")[0] +clu.guid.split("_")[1] + "_" + str(len(list_cluster))
                    list_cluster.append(clu)

            # Add them to the manager and complete them
            for clu in list_cluster:
                leaf_id = self._find_predic_leaf(clu)
                clu.set_prediction(self.dict_label[leaf_id])
                clu.set_expected_prediction_via_proba(self.dict_proba[leaf_id])
                self.manager_cluster.add_cluster(clu)
                
                safe_clu_creater = safe_cluster.SafeCluster(manager_stop=self.manager_stop, cluster_stop=copy.deepcopy(clu),
                                                    dict_leaf_const=self.dict_cst,initial_leaf_id=leaf_id)
                clu_sure = safe_clu_creater.safe_cluster(safety_threshold=0.1)
                if "safe" in clu_sure.guid:
                    self.manager_cluster.add_cluster(clu_sure)

                compl_clu = complete_leaf.CompleteLeaf(manager_stop=self.manager_stop,cluster_stop=copy.deepcopy(clu),
                                                       dict_leaf_const=self.dict_cst,initial_leaf_id=leaf_id)
                new_clu = compl_clu.complete_greedy_cluster(safety_treshold=0.15)
                if "greedy" in new_clu.guid:
                    self.manager_cluster.add_cluster(new_clu)

                switch_clu = switch_leaf.SwitchLeaf(manager_stop=self.manager_stop,cluster_stop=copy.deepcopy(clu),
                                                       dict_leaf_const=self.dict_cst,initial_leaf_id=leaf_id)
                has_decreased = switch_clu.decrease_cluster(safety_threshold=0.1)
                if has_decreased:
                    leaf_id = self._find_predic_leaf(switch_clu.cluster_stop)
                    # assert self.dict_label[leaf_id] < clu.prediction, print(self.dict_label[leaf_id], clu.prediction)
                    switch_clu.cluster_stop.set_prediction(self.dict_label[leaf_id])
                    switch_clu.cluster_stop.set_expected_prediction_via_proba(self.dict_proba[leaf_id])
                    self.manager_cluster.add_cluster(switch_clu.cluster_stop)

        # self.check_accuracy(list(self.manager_cluster.keys()))
        # assert False

    def check_accuracy(self,list_clu):
        """
        Check the clusters accuracy...
        :return:
        """
        accuracy = 0
        num_total_vehicle = 0
        total_dist = 0

        for clus_id in list_clu:
            clus = self.manager_cluster[clus_id]
            routing_solver = cvrptw_routing_solver.RoutingSolverCVRPTW(clus,self.config)
            num_vehicle,distance,list_routes = routing_solver.solve_parse_routing()

            total_dist += distance
            num_total_vehicle += num_vehicle

            # check accuracy
            if num_vehicle == clus.prediction:
                print("accurate ",clus.prediction,clus.expected_prediction, " vs ", num_vehicle, " for ",clus.guid)
                accuracy += 1
            else:
                print("error ",clus.prediction,clus.expected_prediction, " vs ", num_vehicle, " for ",clus.guid)

        accuracy = accuracy/len(list_clu)
        print(" We have an accuracy of ", accuracy, " for total number vehu ", num_total_vehicle, " nad distance ",total_dist)


    def _find_predic_leaf(self,clust):
        """
        Find in which leaf it belongs to
        :param clust: the cluster considered
        :return: a leaf
        """
        object_feature = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(manager_stop=clust)
        dict_feature = object_feature.derive_features()

        for leaf_id in self.dict_cst:
            list_cst = self.dict_cst[leaf_id]
            has_found = True

            for cst in list_cst:
                if not cst.is_respected(dict_feature):
                    has_found = False
                    break

            if has_found:
                return leaf_id

        else:
            assert False


    def kmean_clustering(self,number_cluster):
        """
        Perform a Kmean algorithm on the given stops with a given number of clusters
        :param number_cluster: the number of clusters
        :return: a list of clusters
        """
        matrix_array = np.array([[stop.x,stop.y] for stop in self.manager_stop.values()])

        clusters_list_stop = KMeans(n_clusters = number_cluster).fit_predict(X=matrix_array)
        dict_cluster_stops = dict()
        for i,stop_id in enumerate(list(self.manager_stop.keys())):
            cluster_id = clusters_list_stop[i]

            if not cluster_id in dict_cluster_stops.keys():
                dict_cluster_stops[cluster_id] = cluster.Cluster(depot= self.manager_stop.depot, guid="kmean_" + str(number_cluster) + "_" + str(cluster_id))
            dict_cluster_stops[cluster_id].add_stop(self.manager_stop[stop_id])

        list_cluster = list(dict_cluster_stops.values())
        return list_cluster



if __name__ == '__main__':
    filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/homberger/homberger_600_customer_instances/R1_6_1.TXT"
    man_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    confi = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

    obj_cre = create_all_cluster(man_ref,confi)
    obj_cre.create_all_clusters()
    #obj_cre.check_accuracy()
