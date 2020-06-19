
from src.learning_step.learning_trees import find_best_tree
from src.optimization_step.direct_MIP import clustering_cst_parallel_MIP
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.learning_step.pixelation import pixelManagerAbsolut
from src.helpers import useful_paths,size_data
from src import main_data_creation
from src.data_set_creation.routing_model import cvrptw_routing_solver

import copy

def find_stop_further_away(threshold, manager_stop):

    dict_pair = {}
    for stop_id_1 in manager_stop.keys():
        dict_pair[stop_id_1] = []
        for stop_id_2 in manager_stop.keys():
            dist = manager_stop.matrix_stops_dist[stop_id_1][stop_id_2]
            if dist >= threshold:
                dict_pair[stop_id_1].append(stop_id_2)
    return dict_pair

def find_cluster(dict_cst, dict_leaf_label,manager_pixel,manager_stop,dist_matrix):
    """
    Based on the tree information, derive the best clusters based on Mip for the stop manager
    :param dict_cst: a dict[leaf] = list of cst
    :param dict_leaf_label: a dict[leaf] = label
    :param manager_pixel: a manager pixel
    :param manager_stop: a manager stop
    :return: a dict[leaf] = list of stops
    """
    # need to duplicate the leaves.
    number_copy = 40
    new_dict_cst = {}
    new_dict_label = {}
    for leaf_id in dict_leaf_label:
        orginal_cst = copy.deepcopy(dict_cst[leaf_id])
        original_label = dict_leaf_label[leaf_id]

        for co in range(0,number_copy):
            new_id = leaf_id +"_co_" + str(co)
            new_dict_cst[new_id] = orginal_cst
            if original_label <=3:
                new_dict_label[new_id] = original_label
            else:
                new_dict_label[new_id] = 1000   # we penalized them

    # find pair above treshold
    dict_too_far_pair = find_stop_further_away(threshold=50,manager_stop=manager_stop)

    obj_solver = clustering_cst_parallel_MIP.ClusteringCstParallelMIP(
                                                                      manager_stop=manager_stop,
                                                                      dict_leaf_cst=new_dict_cst,
                                                                      dict_leaf_label= new_dict_label)

    dict_leaf_stop = obj_solver.solve()

    return dict_leaf_stop,new_dict_label

def find_cluster_created(dict_cst, dict_leaf_label,manager_stop):
    """
    Based on the tree information, derive the best clusters based on Mip for the stop manager
    :param dict_cst: a dict[leaf] = list of cst
    :param dict_leaf_label: a dict[leaf] = label
    :param manager_pixel: a manager pixel
    :param manager_stop: a manager stop
    :return: a dict[leaf] = list of stops
    """
    # need to duplicate the leaves.
    number_copy = 40
    new_dict_cst = {}
    new_dict_label = {}
    for leaf_id in dict_leaf_label:
        orginal_cst = copy.deepcopy(dict_cst[leaf_id])
        original_label = dict_leaf_label[leaf_id]

        for co in range(0,number_copy):
            new_id = leaf_id +"_co_" + str(co)
            new_dict_cst[new_id] = orginal_cst
            if original_label <=4:
                new_dict_label[new_id] = original_label
            else:
                new_dict_label[new_id] = 1000   # we penalized them

    # find pair above treshold
    # dict_too_far_pair = find_stop_further_away(threshold=50,manager_stop=manager_stop)

    obj_solver = clustering_cst_parallel_MIP.ClusteringCstParallelMIP(manager_stop=manager_stop,
                                                                      dict_leaf_cst=new_dict_cst,
                                                                      dict_leaf_label= new_dict_label)

    dict_leaf_stop = obj_solver.solve()

    return dict_leaf_stop,new_dict_label

def route_clusters(dict_cluster_stop, manager_stop,dict_leaf_label,config_object):
    """
    Rout the obtained cluster to obtain the final solution
    :param dict_cluster_stop: a dict[leaf_id] = list_stops
    :param manager_stop: a manager stop
    :return: a dict of list of routes for each cluster
    """
    accuracy = 0
    tot_num_vehicles = 0
    total_distance = 0
    dict_routes = {}
    for cluster_id in dict_cluster_stop.keys():
        cluster_manager_stop = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(dict_cluster_stop[cluster_id],manager_stop)
        routing_solver = cvrptw_routing_solver.RoutingSolverCVRPTW(cluster_manager_stop,config_object)
        num_vehicle,distance,list_routes = routing_solver.solve_parse_routing()
        dict_routes[cluster_id] = list_routes
        tot_num_vehicles += num_vehicle
        total_distance += distance

        # check accuracy
        if num_vehicle == dict_leaf_label[cluster_id]:
            accuracy += 1

    accuracy = accuracy/len(dict_cluster_stop)
    print(" We have an accuracy of ", accuracy, " a total number of vehicles ", tot_num_vehicles, " and a total distance ",total_distance)


def solve_routing_problem(filename):
    """
    Solve the routing problem on this filename
    :param filename: the name of the file
    :return: the solution of the large scale routing problem.
    """
    config_object = main_data_creation.read_config(filename)
    manager_stop = main_data_creation.read_stop(filename,date=None)

    best_tree_finder = find_best_tree.FindBestTree()
    learner, dict_constraint, dict_label, dict_proba = best_tree_finder.find_cst_best_tree()

    dict_leaf_stop,new_dict_label = find_cluster_created(dict_constraint, dict_label,manager_stop)

    route_clusters(dict_leaf_stop,manager_stop,new_dict_label,config_object)


if __name__ == '__main__':
    filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created/1000_customers/test_9.txt"
    solve_routing_problem(filename)

