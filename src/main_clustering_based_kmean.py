
from src.optimization_step.creation_shape_non_linear.Kmean_based import create_all_clusters
from src.optimization_step.creation_shape_non_linear.Kmean_based import final_set_covering_pb
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw
from src.data_set_creation.routing_model import cvrptw_routing_solver



if __name__ == '__main__':
    filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/600_customers/test_9.txt"
    man_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    confi = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

    # # benchmark Local Solver
    # router = cvrptw_routing_solver.RoutingSolverCVRPTW(man_ref,confi)
    # num_vehicle,tot_distance,list_routes = router.solve_parse_routing()
    # print("Benchmark ", num_vehicle, " vehicles for total dist ",tot_distance)

    obj_cre = create_all_clusters.create_all_cluster(man_ref,confi)
    obj_cre.create_all_clusters()
    # obj_cre.check_accuracy()

    dict_clus_stop = {clus_id : list(clu.keys()) for clus_id, clu in obj_cre.manager_cluster.items()}
    dict_clus_prediction = {clus_id : clu.expected_prediction for clus_id, clu in obj_cre.manager_cluster.items()}
    for clus_id in dict_clus_prediction:
        if dict_clus_prediction[clus_id] >=4.5:
            dict_clus_prediction[clus_id] = 1000

    mip_cover = final_set_covering_pb.MIP_set_covering(list_stop= list(man_ref.keys()),dict_clus_stop=dict_clus_stop,
                                                       dict_clus_predict=dict_clus_prediction)
    list_final_clus = mip_cover.solve()
    # dict_final_clus = {clus_id : obj_cre.manager_cluster[clus_id] for clus_id in list_final_clus}

    obj_cre.check_accuracy(list_final_clus)


