
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw

from src.data_set_creation.routing_model import cvrptw_routing_solver
from src.helpers import size_data,logging_files
from src.optimization_step.scp_approach.objects import cluster
from src.learning_step.learning_trees import find_best_tree
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset



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

    router = cvrptw_routing_solver.RoutingSolverCVRPTW(manager_stop,config,time_routing=600)
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
    return num_vehicle,'LocalSolver'



if __name__ == '__main__':
    # reset logs
    logging_files.reset_all()

    filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/1500_customers/test_9.txt"
    size_data.NUMBER_CUSTOMERS = 1500
    man_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    confi = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

    # solve via LocalSolver
    tot_vehi,type_benchmark = solve_whole_local_solver(manager_stop=man_ref,config=confi)




