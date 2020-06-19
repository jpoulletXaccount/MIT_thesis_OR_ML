
import os,sys,time,random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Data creation
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw
from src.data_set_creation.clustering_process import perform_clustering
from src.data_set_creation.routing_model import cvrptw_routing_solver

# Learning
from src.learning_step.learning_trees import find_best_tree
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset

# Optimization
# from src.optimization_step.creation_shape_non_linear.Kmean_based import create_all_clusters,final_set_covering_pb
from src.optimization_step.scp_approach import main_clustering_algo
from src.optimization_step.scp_approach.objects import cluster


from src.helpers import useful_paths,local_path
from src.helpers.logging_files import coordinator_logger

class CoordinatorInterface(object):
    """
    Class which coordonne the learning with the routing, main class.
    """

    def __init__(self,list_filename, number_data_points):
        self.list_filename = list_filename
        self.tree_tuner = find_best_tree.FindBestTree()

        # To be updated at each iteration
        self.current_file = None
        self.manager_ref = None
        self.config_ref = None
        self._update_config_manager_file()

        # Params
        self.number_data_points = number_data_points
        self.iteration = 0


        # Stats
        self.list_acc_trees_train = []
        self.list_acc_trees_test = []
        self.predicted_expected_number_vehi = []
        self.predicted_nb_vehi = []
        self.list_acc_mip = []
        self.obtained_number_vehi = []
        self.list_number_instances = []


    def perform_creation_based_trees(self,from_points):
        """
        Main function, perform the creation based on the trees sequentially obtained
        :param from_points: boolean indicating if we should consider the points already gathered or not
        :return:
        """

        if not from_points:
            self.iteration = 0
            # Clean all files
            list_file = [useful_paths.TREE_JSON_CVRPTW, useful_paths.FILE_TO_STORAGE_CVRPTW, useful_paths.STATS_LEARN_OPTIMIZE, useful_paths.FILE_TO_TREE_CVRPTW]
            self._remove_file(list_file)

            self._initialize_data_point("KMEAN")
            self._initialize_data_point("RANDOM")

        else:
            # Need to find the iteraiton number
            self._re_update_trackers()
            self.iteration = len(self.obtained_number_vehi)

        go_on = True
        while go_on:

            # Compute the current trees
            tree, dict_cst, dict_leaf_label, dict_proba,dict_disper = self._compute_OCT_tree()

            # Update filename
            self._update_config_manager_file()

            # Create potential set and add points
            obj_main_clustering = self._create_potential_clusters(tree,dict_cst,dict_disper)
            self._find_most_relevant_ones(obj_main_clustering)

            # Check if we add enough points
            df_points = pd.read_csv(useful_paths.FILE_TO_STORAGE_CVRPTW)
            self.list_number_instances.append(len(df_points))
            go_on = len(df_points) <= self.number_data_points

            # Indicates progress
            sys.stdout.write("we have gathered: " + str(len(df_points)))
            sys.stdout.write("\n")
            sys.stdout.flush()

            # output
            self._output()

            # save stats main clustering
            stats_source = local_path.PATH_TO_PROJECT + 'database/cvrptw/stats_algo.csv'
            stats_dest = local_path.PATH_TO_PROJECT + 'database/cvrptw/stats_algo_' + str(self.iteration) + '.csv'
            os.rename(stats_source,stats_dest)
            self.iteration +=1

        self._output()


    def _re_update_trackers(self):
        """
        In a case of re-updating
        :return:
        """
        pd_stats = pd.read_csv(useful_paths.STATS_LEARN_OPTIMIZE)
        self.list_acc_trees_train = list(pd_stats['train_acc'])
        self.list_acc_trees_test = list(pd_stats['test_acc'])
        self.list_acc_mip = list(pd_stats['mip_acc'])
        self.predicted_expected_number_vehi = list(pd_stats['predicted_expected_vehi'])
        self.predicted_nb_vehi = list(pd_stats["predicted_nb_vehi"])
        self.obtained_number_vehi = list(pd_stats["obtaind_vehi"])
        self.list_number_instances = list(pd_stats['size_instance'])



    def _output(self):
        """
        Output all the stats under a json dict
        :return:
        """
        self._remove_file([useful_paths.STATS_LEARN_OPTIMIZE])

        final_data = []
        for i in range(0,len(self.list_acc_mip)):
            row = dict()
            row["train_acc"] = self.list_acc_trees_train[i]
            row["test_acc"] = self.list_acc_trees_test[i]
            row["mip_acc"] = self.list_acc_mip[i]
            row["predicted_expected_vehi"] = self.predicted_expected_number_vehi[i]
            row["predicted_nb_vehi"] = self.predicted_nb_vehi[i]
            row["obtaind_vehi"] = self.obtained_number_vehi[i]
            row["size_instance"] = self.list_number_instances[i]
            final_data.append(row)

        # Dump
        final_data =pd.DataFrame(final_data)
        final_data.to_csv(useful_paths.STATS_LEARN_OPTIMIZE, header=True)


    def _update_config_manager_file(self):
        """
        At each iteration update that
        """
        self.current_file = self._choose_filename()
        self.manager_ref, self.config_ref = self._get_manager_config(self.current_file)


    def _initialize_data_point(self, cluster_method):
        """
        Create the first data points and write them
        :return:
        """
        self.config_ref.cluster_method = cluster_method

        clustering_object = perform_clustering.StopClustering()
        list_cluster = clustering_object.perform_clustering(self.manager_ref,self.config_ref)

        list_manager = []
        for cluster in list_cluster:
            cluster.depot = self.manager_ref.depot
            cluster.matrix_depot_dist = self.manager_ref.matrix_depot_dist
            cluster.matrix_stops_dist = self.manager_ref.matrix_stops_dist
            current_manager_stops = stops_manager_cvrptw.StopsManagerCVRPTW.init_from_cluster(cluster)
            current_manager_stops.set_depot(self.manager_ref.depot)
            list_manager.append(current_manager_stops)

        list_vehi =self._add_new_data_points(list_manager)
        self._add_new_features(list_manager,list_vehi)

        coordinator_logger.info('Init done for ' + str(classmethod))

    def _compute_OCT_tree(self):
        """
        Compute the best OCT tree and output it
        :return: dict_cst, dict_leaf_label, dict_proba
        """
        self._remove_file([useful_paths.TREE_JSON_CVRPTW])
        self.tree_tuner.set_file_and_df(file=useful_paths.FILE_TO_TREE_CVRPTW)
        tree, dict_cst, dict_leaf_label, dict_proba,dict_features_dispersion = self.tree_tuner.find_cst_best_tree()

        self.list_acc_trees_test.append(self.tree_tuner.score_test)
        self.list_acc_trees_train.append(self.tree_tuner.score_train)

        return tree, dict_cst, dict_leaf_label, dict_proba, dict_features_dispersion


    def _create_potential_clusters(self,tree,dict_cst,dict_dispersion):
        """
        Create all potential relevant clusters, based on the given trees
        :return: the object used in the main clustering algo
        """
        self.config_ref.cluster_method = "OPTIM"
        #obj_cre = create_all_clusters.create_all_cluster(self.manager_ref,self.config_ref,dict_cst, dict_leaf_label, dict_proba)
        # obj_cre.create_all_clusters()
        obj_cre = main_clustering_algo.main_clustering_algo(self.manager_ref,self.config_ref,tree,dict_cst,dict_dispersion)
        obj_cre.create_relevant_clusters()

        return obj_cre



    def _find_most_relevant_ones(self,obj_main_clustering):
        """
        Among all the potential cluster, choose the best ones and add them to the data points
        :param obj_main_clustering: the object for the main clustering considered
        :return:
        """
        all_cluster_id = []

        # take the optimal clusters in the linear relaxation
        time_begin = time.time()
        _, dict_reduced_cost, _, _,dict_x = obj_main_clustering.solve_scp(relax = True)
        cluster_relaxed = [clu_id for clu_id,rc in dict_reduced_cost.items() if rc ==0]
        x_threshold = np.percentile([dict_x[clu_id] for clu_id in cluster_relaxed],75)
        cluster_relaxed = [clu_id for clu_id in cluster_relaxed if dict_x[clu_id] >= x_threshold]
        cluster_relaxed.sort(reverse=True)
        time_relax = time.time()


        # We are going to loop 2 times to get the most relevant clusters
        for i in range(0,2):

            # Check if feasible problem...
            list_nb = [len(list_covertness) for list_covertness in obj_main_clustering.manager_cluster.dict_stop_clusters.values()]
            if min(list_nb) <= 0.5:
                coordinator_logger.debug('Not enough clusters to cover all stops at iteration ', i)
                break

            list_selected_clusters, _,_,obj_val,_ = obj_main_clustering.solve_scp(relax=False)
            assert len([clu_id for clu_id in list_selected_clusters if clu_id in all_cluster_id]) == 0, print(list_selected_clusters,all_cluster_id)
            all_cluster_id.extend(list_selected_clusters)
            final_list_cluster = [obj_main_clustering.manager_cluster[clu_id] for clu_id in list_selected_clusters]
            coordinator_logger.info('At iteration of the solve ' + str(i) + ' we have selected ' + str(len(list_selected_clusters)) + ' and obj val ' + str(obj_val))
            stats_vehi = False
            if i == 0:
                expected_number_vehi = sum(clu.expected_prediction for clu in final_list_cluster)
                self.predicted_expected_number_vehi.append(expected_number_vehi)
                self.predicted_nb_vehi.append(obj_val)
                stats_vehi = True

            list_vehi = self._add_new_data_points(list_manager_stop=final_list_cluster,stats=stats_vehi)
            self._add_new_features(final_list_cluster,list_vehi)

            # We remove the one used and continue, to get sligthly more data point faster
            for clus_id in list_selected_clusters:
                obj_main_clustering.manager_cluster.delete_cluster(clus_id)

        time_end = time.time()
        coordinator_logger.info('Number of clusters from the integer problem '+ str(len(all_cluster_id)) +  ' in '+ str(time_end-time_relax) + 's.')

        # Deal with the cluster coming from the relaxation and not seen in the interger solution
        cluster_relaxed = [clu_id for clu_id in cluster_relaxed if not clu_id in all_cluster_id]
        random.shuffle(cluster_relaxed)
        # find the one corresponding to 4
        cluster_four = [clu_id for clu_id in cluster_relaxed if obj_main_clustering.manager_cluster[clu_id].prediction == 4]
        coordinator_logger.info('Number of cluster four total ' +str(len(cluster_four)))
        selected_four = cluster_four[0:min(len(cluster_four),round(0.5 *len(all_cluster_id)))]

        cluster_relaxed = [clu_id for clu_id in cluster_relaxed if not clu_id in selected_four]
        cluster_relaxed = cluster_relaxed[0:min(len(cluster_relaxed),round(0.5 *len(all_cluster_id)))]   # We make sure that not too many cluster from the relaxation...
        cluster_relaxed = [obj_main_clustering.manager_cluster[clu_id] for clu_id in cluster_relaxed]
        cluster_relaxed.extend([obj_main_clustering.manager_cluster[clu_id] for clu_id in selected_four])
        list_vehi = self._add_new_data_points(list_manager_stop=cluster_relaxed,stats=False)
        self._add_new_features(cluster_relaxed,list_vehi)
        coordinator_logger.info('Number clusters from the relaxation '+ str(len(cluster_relaxed)) + ' in '+ str(time_relax - time_begin + time.time() - time_end) + 's.')


    def _add_new_data_points(self,list_manager_stop,stats = False):
        """
        Solve the manager stops in the given list and ouput the results in a file
        :param list_manager_stop: a list of manager stop object
        :param stats: indicates if we stock the stats or not
        :return: the list of vehicles, in the same order as the manager stop
        """
        list_vehi = []
        tot_vehi = 0
        acc = 0
        for manager_stop in tqdm(list_manager_stop, desc= 'solving the true routing'):
            solver_obj = cvrptw_routing_solver.RoutingSolverCVRPTW(manager_stop, self.config_ref)
            num_vehicle,tot_distance,list_routes = solver_obj.solve_routing()
            tot_vehi += num_vehicle
            list_vehi.append(num_vehicle)

            if isinstance(manager_stop,cluster.Cluster):
                if manager_stop.prediction == num_vehicle:
                    coordinator_logger.info('Accurate ' + str(manager_stop.prediction) + ' ' + str(manager_stop.expected_prediction) + ' vs ' + str(num_vehicle)+ ' for cluster ' + manager_stop.guid)
                    acc +=1
                else:
                    coordinator_logger.info('Error ' + str(manager_stop.prediction) +' ' + str(manager_stop.expected_prediction) + ' vs ' + str(num_vehicle)+ ' for cluster ' + manager_stop.guid)

        if stats:
            self.obtained_number_vehi.append(tot_vehi)
            self.list_acc_mip.append(acc/len(list_manager_stop))

        coordinator_logger.info('Overall accuracy ' + str(acc/len(list_manager_stop)))

        return list_vehi


    def _add_new_features(self,list_manager_stop,list_vehi):
        """
        Compute the features of the manager stops to be added
        :param list_manager_stop: a list of manager stop object
        :param list_vehi: a list of number of vehicle obtained via the routing
        :return:
        """
        data = []

        for i,manager_stop in enumerate(list_manager_stop):
            feature_object = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(manager_stop)
            dict_features= feature_object.derive_features()
            dict_features['class_label'] = min(6,list_vehi[i])
            dict_features['iteration'] = self.iteration
            data.append(dict_features)

        header = True
        if os.path.exists(useful_paths.FILE_TO_TREE_CVRPTW):
            header = False

        data_results = pd.DataFrame(data)
        data_results.to_csv(useful_paths.FILE_TO_TREE_CVRPTW, header=header, index=False,mode = "a")


    def _choose_filename(self):
        """
        Randomly choose a filename
        :return: such a filename
        """
        idx = np.random.randint(0,len(self.list_filename))
        return self.list_filename[idx]

    @staticmethod
    def _remove_file(list_file):
        """
        Remove file if exists
        :param list_file: a list of files to remove
        :return:
        """
        for file in list_file:
            if os.path.exists(file):
                os.remove(file)

    @staticmethod
    def _get_manager_config(filename):
        """
        Get the mnaager stop and the config corresponding to the filename
        :param filename: a given filename
        :return: a manager_Stop, a config
        """
        manager_reference = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
        config = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

        return manager_reference,config
