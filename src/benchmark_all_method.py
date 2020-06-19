
import pandas as pd
from tqdm.auto import tqdm
import os,sys
sys.path.append(os.getcwd().split("src")[0])

from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw
from src.data_set_creation.clustering_process import perform_clustering
from src.data_set_creation.routing_model import cvrptw_routing_solver
from src.helpers import size_data,logging_files,local_path,useful_paths
from src.optimization_step.scp_approach import main_clustering_algo
from src.optimization_step.scp_approach.objects import cluster
from src.learning_step.learning_trees import find_best_tree
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset


def solve_via_cluster(manager_stop,config,perce):
    """
    Route the whole manager sotp with respect to the config via naive Kmean
    :param manager_stop:
    :param config:
    :param perce: percentage of the vehicle's capacity.
    :return:
    """
    tree_finder = find_best_tree.FindBestTree()
    dict_feature = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(man_ref)
    tree, dict_cst_computed, dict_leaf_label_computed, dict_proba_computed, dict_dispersion = tree_finder.find_cst_best_tree()
    list_cst = [cst for l in dict_cst_computed.values() for cst in l]
    list_features = [cst.get_feature_name(dict_feature) for cst in list_cst]
    list_features=list(set(list_features))

    config.cluster_method = "NAIVE"
    clustering_object = perform_clustering.StopClustering()
    list_cluster = clustering_object.perform_clustering(manager_stop,config,perce)

    print("Number of clusters ", len(list_cluster))
    tot_vehi = 0
    predicted= 0
    acc = 0

    for clu in tqdm(list_cluster,desc='Routing in naive benchmark'):
        man_stop_clus = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(list_stops=[stop.guid for stop in clu.list_stops],reference_manager_stop=manager_stop)
        cluster_object = cluster.Cluster.from_manager_stops(man_stop_clus,'clu_0',tree,dict_cst_computed,list_features,dict_dispersion)
        predicted += cluster_object.prediction
        current_manager_stops = stops_manager_cvrptw.StopsManagerCVRPTW.init_from_cluster(clu)
        current_manager_stops.set_depot(manager_stop.depot)

        router = cvrptw_routing_solver.RoutingSolverCVRPTW(current_manager_stops,config)
        num_vehicle,tot_distance,list_routes = router.solve_parse_routing()
        tot_vehi += num_vehicle

        if num_vehicle ==  cluster_object.prediction:
            acc +=1

    print("Benchmark naive Kmean", tot_vehi,"for predicted ", predicted, " accuracy ", acc/len(list_cluster))
    return predicted,len(list_cluster),acc/len(list_cluster), tot_vehi,"Kmean"


def solve_whole_local_solver(manager_stop,config):
    """
    Solve the problem as a whole, for the allocated time
    :param manager_stop: the manager stop as a whole
    :param config: the associated config
    :return: tot_vehi
    """
    tree_finder = find_best_tree.FindBestTree()
    dict_feature = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(man_ref)
    tree, dict_cst_computed, dict_leaf_label_computed, dict_proba_computed, dict_dispersion = tree_finder.find_cst_best_tree()
    list_cst = [cst for l in dict_cst_computed.values() for cst in l]
    list_features = [cst.get_feature_name(dict_feature) for cst in list_cst]
    list_features=list(set(list_features))
    router = cvrptw_routing_solver.RoutingSolverCVRPTW(manager_stop,config,time_routing=3600)
    num_vehicle,tot_distance,list_routes = router.solve_parse_routing()

    number_vehicle_predicted = 0
    acc = 0
    for route in list_routes:
        new_manager = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(route,manager_stop)
        new_cluster = cluster.Cluster.from_manager_stops(manager_stop=new_manager,
                                                         guid='clu_0',
                                                        tree=tree,
                                                         dict_cst= dict_cst_computed,
                                                         list_useful_features=list_features,
                                                         dict_disp=dict_dispersion)
        print('For the route we have a prediction of ', new_cluster.prediction)
        number_vehicle_predicted += new_cluster.prediction
        if new_cluster.prediction == 1:
            acc += 1


    print('Benchmark local solver ', num_vehicle, ' for prediction according to tree ', number_vehicle_predicted, ' and accuracy ',acc/len(list_routes))
    return number_vehicle_predicted,len(list_routes),acc/len(list_routes), num_vehicle,"LocalSolver"


def solve_clustering_algo(manager_stop,config):
    """
    Solve the problem using the clustering algo designed
    :param manager_stop: manager stops considered
    :param config: the config associated
    :return:
    """
    object_clustering = main_clustering_algo.main_clustering_algo(manager_stop,config)
    object_clustering.create_relevant_clusters(read_cluster=False)
    cluster_selected = object_clustering.solve()
    print("Number of clusters ", len(cluster_selected))
    tot_vehi = 0
    predicted= 0
    acc = 0
    for clus_id in tqdm(cluster_selected, desc='Routing in clustering benchmark'):
        clus = object_clustering.manager_cluster[clus_id]
        predicted += clus.prediction

        router = cvrptw_routing_solver.RoutingSolverCVRPTW(clus,config)
        num_vehicle,tot_distance,list_routes = router.solve_parse_routing()
        tot_vehi += num_vehicle

        if clus.prediction == num_vehicle:
            acc +=1
    print("Benchmark clustering algo", tot_vehi, " vehicles for predicted ", predicted, " accuracy ", acc/len(cluster_selected))

    return predicted,len(cluster_selected),acc/len(cluster_selected), tot_vehi,"Clustering_algo"


def move_to_basic_scp():
    """
    Do all the renaming so that we could tackle the scp basic approach
    :return:
    """
    # Remove the lo
    tree_source = useful_paths.TREE_JSON_CVRPTW
    dict_disp_source = useful_paths.DISPERSION_FEATURES_JSON
    tree_dest =  useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/tree_cvrptw_l_o.json'
    dict_disp_dest = useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/dispersion_features_l_o.json'
    os.rename(tree_source,tree_dest)
    os.rename(dict_disp_source,dict_disp_dest)

    # Switch to SCA
    tree_dest = useful_paths.TREE_JSON_CVRPTW
    dict_disp_dest = useful_paths.DISPERSION_FEATURES_JSON
    tree_source =  useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/tree_cvrptw_sca.json'
    dict_disp_source = useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/dispersion_features_sca.json'
    os.rename(tree_source,tree_dest)
    os.rename(dict_disp_source,dict_disp_dest)


def check_all_files():

    assert os.path.isfile(useful_paths.TREE_JSON_CVRPTW)
    assert os.path.isfile(useful_paths.DISPERSION_FEATURES_JSON)
    assert os.path.isfile(useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/tree_cvrptw_sca.json')
    assert os.path.isfile(useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/dispersion_features_sca.json')


if __name__ == '__main__':
    # reset logs
    logging_files.reset_all()

    # check_all_files()

    filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/7500_customers/test_9.txt"
    size_data.NUMBER_CUSTOMERS = 7500
    man_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    confi = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

    nb_iteration = 2 # Number of try we allocate to it
    data = []

    # using Kmean
    for i in range(0,nb_iteration):
        predicted_number, nb_clusters, accuracy, tot_vehi, type_benchmark = solve_via_cluster(manager_stop=man_ref,config=confi,perce=1)
        row = dict()
        row['predicted_number'] = predicted_number
        row['total_nb_clusters'] = nb_clusters
        row['accuracy'] = accuracy
        row['total_vehi'] = tot_vehi
        row['Benchmark'] = type_benchmark
        data.append(row)
    #
    # # using Clustering algo based on Learn and optimize
    # for i in range(0,nb_iteration):
    #     predicted_number, nb_clusters, accuracy, tot_vehi, _ = solve_clustering_algo(manager_stop=man_ref,config=confi)
    #     row = dict()
    #     row['predicted_number'] = predicted_number
    #     row['total_nb_clusters'] = nb_clusters
    #     row['accuracy'] = accuracy
    #     row['total_vehi'] = tot_vehi
    #     row['Benchmark'] = 'Learn_optimize'
    #     data.append(row)
    #
    # # prepare for change
    # move_to_basic_scp()
    #
    # using Clustering algo based on basic SCP
    # for i in range(0,nb_iteration):
    #     predicted_number, nb_clusters, accuracy, tot_vehi, _ = solve_clustering_algo(manager_stop=man_ref,config=confi)
    #     row = dict()
    #     row['predicted_number'] = predicted_number
    #     row['total_nb_clusters'] = nb_clusters
    #     row['accuracy'] = accuracy
    #     row['total_vehi'] = tot_vehi
    #     row['Benchmark'] = 'Basic_SCA'
    #     data.append(row)

    # solve via LocalSolver
    # predicted_number, nb_clusters, accuracy, tot_vehi,type_benchmark = solve_whole_local_solver(manager_stop=man_ref,config=confi)
    # row = dict()
    # row['predicted_number'] = predicted_number
    # row['total_nb_clusters'] = nb_clusters
    # row['accuracy'] = accuracy
    # row['total_vehi'] = tot_vehi
    # row['Benchmark'] = type_benchmark
    # data.append(row)


    data = pd.DataFrame(data)
    data.to_csv('/Users/jpoullet/Documents/MIT/Thesis/database/cvrptw/benchmark_results.csv',index=False)




