
import pandas as pd
from tqdm.auto import tqdm
import os,math

from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw

from src.data_set_creation.routing_model import cvrptw_routing_solver
from src.helpers import size_data
from src.optimization_step.scp_approach import main_clustering_algo




def solve_whole_local_solver(manager_stop,config):
    """
    Solve the problem as a whole, for the allocated time
    :param manager_stop: the manager stop as a whole
    :param config: the associated config
    :return: tot_vehi
    """
    router = cvrptw_routing_solver.RoutingSolverCVRPTW(manager_stop,config,time_routing=1000)
    num_vehicle,tot_distance,list_routes = router.solve_parse_routing()

    print('Benchmark local solver ', num_vehicle)
    return num_vehicle,'LocalSolver'


def solve_clustering_algo(manager_stop,config,list_running_time):
    """
    Solve the problem using the clustering algo designed
    :param manager_stop: manager stops considered
    :param config: the config associated
    :param list_running_time: the list of test time to run local solver
    :return:
    """
    object_clustering = main_clustering_algo.main_clustering_algo(manager_stop,config)
    object_clustering.create_relevant_clusters(read_cluster=False)
    cluster_selected = object_clustering.solve()
    print("Number of clusters ", len(cluster_selected))

    data = []
    for ti in list_running_time:
        tot_vehi = 0
        predicted= 0
        acc = 0
        row = dict()
        for clus_id in tqdm(cluster_selected, desc='Routing in clustering benchmark'):
            clus = object_clustering.manager_cluster[clus_id]
            predicted += clus.prediction

            router = cvrptw_routing_solver.RoutingSolverCVRPTW(clus,config,time_routing=ti)
            num_vehicle,tot_distance,list_routes = router.solve_parse_routing()
            tot_vehi += num_vehicle

            if clus.prediction == num_vehicle:
                acc +=1

        row['nb_cluster'] = len(cluster_selected)
        row['indiv_time'] = ti
        row['accuracy'] = acc/len(cluster_selected)
        row['total_nb_veji'] = tot_vehi
        row['predicted_number'] = predicted
        for a in [2,4,6,8]:
            q = math.ceil(len(cluster_selected)/a)
            row['parall_'+str(a)] = q * ti

        data.append(row)

    data = pd.DataFrame(data)
    data.to_csv('/Users/jpoullet/Documents/MIT/Thesis/database/cvrptw/benchmark_running_time.csv',index=False)



if __name__ == '__main__':

    filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/2000_customers/test_9.txt"
    size_data.NUMBER_CUSTOMERS = 2000
    man_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    confi = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

    list_indiv_time = [1,2,3,4,5,7,10,15,20]
    solve_clustering_algo(manager_stop=man_ref,config=confi,list_running_time=list_indiv_time)

    # solve via LocalSolver
    tot_vehi_ls,type_benchmark = solve_whole_local_solver(manager_stop=man_ref,config=confi)



