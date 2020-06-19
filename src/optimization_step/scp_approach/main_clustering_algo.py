
import numpy as np
import pandas as pd
import time,smtplib,ssl,sys,os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tqdm.auto import tqdm

sys.path.append(os.getcwd().split("src")[0])

from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.config import config_constraint_cvrptw
from src.data_set_creation.routing_model import cvrptw_routing_solver

from src.optimization_step.scp_approach.objects import manager_clusters
from src.optimization_step.scp_approach import scp_solver
from src.optimization_step.scp_approach.heuristics_improvement import modify_clusters, merge_clusters, create_clusters
from src.optimization_step.scp_approach.initial_solution import basic_K_mean

from src.learning_step.learning_trees import find_best_tree

from src.helpers import local_path,size_data,logging_files
from src.helpers.logging_files import clustering_logger

class main_clustering_algo(object):
    """
    Main class for the clustering algorithm, solve the clustering problem on a set of stops, using
    a set covering approach in which clusters are iteratively created.
    """

    def __init__(self, manager_stops, config, tree = None, dict_cst= None, dict_dispersion = None):
        self.manager_stops = manager_stops
        self.config = config            # a config object
        if tree is None or dict_cst is None or dict_dispersion is None:
            finder = find_best_tree.FindBestTree()
            self.tree,self.dict_cst, dict_label, _, self.dict_dispersion= finder.find_cst_best_tree()
        else:
            self.tree = tree    # tree object, from the interpretable ai interface.
            self.dict_cst = dict_cst        # a dict[index_leaf] = list of constraints
            self.dict_dispersion = dict_dispersion  # a dict[feature] = faro dispersion

        # To be filled
        self.manager_cluster = manager_clusters.ManagerCluster(manager_stops,self.tree,self.dict_cst,self.dict_dispersion)

        # Parameters hard coded
        dict_param = {600:(20,25,3,65,20),1000:(40,30,4,70,25),1500:(40,35,4,80,30),2000:(40,35,5,85,35),
                        3000:(55,40,6,90,40),5000:(60,45,6,100,45),7500:(60,45,7,100,45),10000:(40,45,7,100,45)}
        self.nb_iter,self.max_clusters_improved,self.number_creation,self.number_considered_merge,self.number_treated_merge = dict_param[size_data.NUMBER_CUSTOMERS]
        self.threshold = 0.0

         # Heuristics
        self.cluster_operation = modify_clusters.modify_clusters(self.manager_cluster,self.manager_stops,self.tree,self.dict_cst,self.threshold)
        self.cluster_merger = merge_clusters.ClustersMerger(self.manager_cluster, self.manager_stops,self.tree, nb_considered= self.number_considered_merge, nb_treated=self.number_treated_merge)
        self.cluster_creation = create_clusters.CreateClusters(self.manager_cluster, self.manager_stops,self.config,self.tree,self.dict_cst)

        # tracking
        self.iteration = 0
        self.total_time_creation = 0
        self.total_time_merge = 0
        self.total_time_improv = 0

        # stats
        self.lp_value = []
        self.nb_total_clusters = []
        self.per_negative_rc = []
        self.avg_negative_rc = []
        self.avg_robustness_created = []
        self.accuracy = []
        self.real_nb_vehi = []
        self.predicted_nb_vehi = []


    def create_relevant_clusters(self, read_cluster = False):
        """
        Main function, create the clusters and solve the scp
        :param read_cluster: True then read cluster previously solved
        :return: the final set of clusters, on which to perform the routing
        """
        self._initialization(read_cluster)

        list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val,dict_x = self.solve_scp(relax=True)
        clustering_logger.info('SCP solve iteration: 0, number of clusters selected ' + str(len(list_selected_clusters)))
        # for clu_id in list_selected_clusters:
        #     print(clu_id,self.manager_cluster[clu_id].expected_prediction, self.manager_cluster[clu_id].tracking_evolution)
        if not read_cluster:
            self._track_stats(obj_val,check_acc=False)
            self.save_results()

        initial_iter = self.iteration + 1
        for it in tqdm(range(initial_iter,self.nb_iter),desc='Iteration of the main loop in clustering algo'):
            self.iteration = it
            self._reset_stats()
            time_begin = time.time()
            self._create_new_clusters(dict_reduced_cost,dict_dual_val)
            time_create = time.time()
            clustering_logger.info('Clusters created in ' + str(time_create - time_begin))
            self.total_time_creation += time_create - time_begin
            self._improve_clusters(dict_x,dict_reduced_cost,dict_dual_val)
            time_improve = time.time()
            clustering_logger.info('Improvement of clusters done in ' + str(time_improve - time_create))
            self.total_time_improv += time_improve - time_create
            self._merge_clusters(dict_x,dict_reduced_cost,dict_dual_val)
            clustering_logger.info('Clusters have been merged in ' + str(time.time() - time_improve))
            self.total_time_merge += time.time() - time_improve

            # rc_negative = self.cluster_operation.reduced_cost_negative
            # print("Number of clusters created with negative reduced cost ", rc_negative, ' on a total of modified ',
            #       self.cluster_operation.total_modify, ' i.e. ', rc_negative/self.cluster_operation.total_modify)

            list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val,dict_x = self.solve_scp(relax=True)
            clustering_logger.info('SCP solve iteration: ' + str(it) + ' number of clusters selected '+ str(len(list_selected_clusters)))
            # for clu_id in list_selected_clusters:
            #     print(clu_id, self.manager_cluster[clu_id].expected_prediction, self.manager_cluster[clu_id].tracking_evolution)

            self._track_stats(obj_val,check_acc=False)
            self.save_results()
            clustering_logger.info('finished iteration in ' + str(time.time() - time_begin))


    def solve(self):
        """
        Solve the integer scp problem with the current clusters in the manager clusters
        :return: the list of selected clusters
        """
        list_selected_clusters, _,_,_,_ = self.solve_scp(relax=False)
        # for clu_id in list_selected_clusters:
        #     print(clu_id, self.manager_cluster[clu_id].expected_prediction, self.manager_cluster[clu_id].tracking_evolution)

        list_robus = [self.manager_cluster[clu_id].is_robust(self.threshold) for clu_id in list_selected_clusters]
        # print('Average robustness ' + str(np.mean(list_robus)))
        return list_selected_clusters


    def _initialization(self,read_cluster):
        """
        Take care of the initialization of the clusters
        :param read_cluster: indicates if we read the cluster from a file or not
        :return: update the manager cluster and the iteration
        """
        file_cluster = local_path.PATH_TO_PROJECT + 'database/cvrptw/clusters_save.csv'
        stats_file = local_path.PATH_TO_PROJECT + 'database/cvrptw/stats_algo.csv'

        # update cluster and stats
        if read_cluster:
            pd_cluster = pd.read_csv(file_cluster)
            self.manager_cluster.build_from_df(df=pd_cluster,manager_ref=self.manager_stops)
            clustering_logger.info('manager built')

            pd_stats = pd.read_csv(stats_file)
            self.iteration = pd_stats['iteration'].max()
            self.lp_value = list(pd_stats['lp_value'])
            self.nb_total_clusters = list(pd_stats['nb_clusters'])
            self.per_negative_rc = list(pd_stats['per_neg_rc'])
            self.avg_negative_rc = list(pd_stats['avg_neg_rc'])
            self.avg_robustness_created = list(pd_stats['avg_robustness_created'])
            self.predicted_nb_vehi = list(pd_stats['predicted_nb_vehi'])
            self.accuracy = list(pd_stats['sca_accuracy'])
            self.real_nb_vehi = list(pd_stats['real_nb_vehi'])

        else:
            self._initialize_clusters()

            if os.path.isfile(file_cluster):
                os.remove(file_cluster)
            if os.path.isfile(stats_file):
                os.remove(stats_file)

        clustering_logger.info('Init done '+ str(len(self.manager_cluster)))
        assert self.manager_cluster.check_cluster_initialized()



    def _initialize_clusters(self):
        """
        Create the first initial clusters
        Update self.manager_clusters
        """
        max_cap = self.config.capacity_cst
        total_demand = self.manager_stops.demand
        list_number_cluster = [int(total_demand/(i * max_cap)) for i in [0.75,1,1.25]]
        # list_number_cluster = [int(total_demand/(k * max_cap)) for k in [0.4]]

        Kmean_basic = basic_K_mean.basicKMeans(manager_cluster=self.manager_cluster,manager_stops=self.manager_stops)
        for k in list_number_cluster:
            Kmean_basic.run_K_mean(list(self.manager_stops.keys()),k)


    def _reset_stats(self):
        """
        Reset the stats of the heuristics
        """
        self.cluster_operation.reset_stats()


    def _track_stats(self,lp_val,check_acc = False):
        """
        Update the stats for each iteration
        """
        self.lp_value.append(lp_val)
        self.nb_total_clusters.append(len(self.manager_cluster))

        if self.cluster_operation.total_modify == 0:
            self.per_negative_rc.append(0)
            self.avg_negative_rc.append(0)
            self.avg_robustness_created.append(0)
        else:
            rc_negative = self.cluster_operation.reduced_cost_negative
            self.per_negative_rc.append(rc_negative/self.cluster_operation.total_modify)
            self.avg_negative_rc.append(np.mean(self.cluster_operation.avg_negative_rc))
            self.avg_robustness_created.append(np.mean(self.cluster_operation.is_robust))

        if check_acc and self.iteration % 5 ==0:
            list_selected_clusters, _, _,obj_val,_ = self.solve_scp(relax=False)
            self.predicted_nb_vehi.append(obj_val)
            acc,total_nb = self.check_accuracy(list_selected_clusters)
            self.accuracy.append(acc)
            self.real_nb_vehi.append(total_nb)
        else:
            self.predicted_nb_vehi.append(0)
            self.accuracy.append(0)
            self.real_nb_vehi.append(0)


    def solve_scp(self, relax):
        """
        Solve the set covering problem with the clusters in the current manager
        :param relax: indicates if we solve the relaxed version of the scp
        :return: list of selected cluster, dict of reduced cost and dual values, objective value, dict of x value
        """

        if relax:
            return self._solve_relax_scp()

        else:
            return self._solve_integer_scp()


    def _solve_relax_scp(self):
        """
        Solve the relaxation of the scp
        :return:list of selected cluster, dict of reduced cost and dual values, objective value, dict of x value
        """
        for clu_id, clu in self.manager_cluster.items():
            if not clu.is_robust(self.threshold) and clu.prediction >= 3.5:
                clu.prediction = 10000
                clu.expected_prediction = 10000
        scp_mip = scp_solver.MIP_set_covering(list_stop=list(self.manager_stops.keys()),
                                              dict_stop_clus= self.manager_cluster.dict_stop_clusters,
                                              dict_clus_predict= {clu_id : clu.expected_prediction for clu_id, clu in self.manager_cluster.items()})

        list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val, dict_x = scp_mip.solve(relax=True,warm_start=None)

        return list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val,dict_x


    def _solve_integer_scp(self):
        """
        solve the integer version of the scp
        :return: list of selected cluster, dict of reduced cost and dual values, objective value, dict of x value
        """
        time_begin = time.time()
        _, dict_reduced_cost, _, lb,dict_x = self._solve_relax_scp()
        time_relax = time.time()
        clustering_logger.info('Lower bound on the relaxation ' + str(lb) + ' in ' + str(time_relax - time_begin) + ' total nb clusters '+ str(len(self.manager_cluster)))

        ub_branching,final_clusters = self._upper_bound_by_branching(dict_x)
        assert ub_branching >= lb-0.01, print(ub_branching,lb)


        rc_threshold = np.percentile(list(dict_reduced_cost.values()),5)
        rc_threshold = max(0,rc_threshold)
        list_considered_clusters = [clu_id for clu_id, val in dict_reduced_cost.items() if val <= rc_threshold]
        list_considered_clusters.extend(final_clusters)
        list_considered_clusters = list(set(list_considered_clusters))
        is_covered = [stop_id for clu_id in list_considered_clusters for stop_id in self.manager_cluster[clu_id]]
        assert len(set(self.manager_stops.keys()).difference(set(is_covered))) == 0, print(set(self.manager_stops.keys()).difference(set(is_covered)))
        list_selected_clusters, _, _, ub_restricted, _ = self._solve_scp_resctricted_cluster(list_considered_clusters,relax=False,time_run=180,warm_start=final_clusters)
        time_ub = time.time()
        clustering_logger.info('Upper bound on the relaxation ' + str(ub_restricted) + ' threshold used '+ str(rc_threshold) + ' leading to ' + str(len(list_considered_clusters)) + ' solved in ' + str(time_ub - time_relax))

        ub = min(ub_restricted,ub_branching)

        # filter only on the one with a low reduced cost
        list_considered_clusters = [clu_id for clu_id, val in dict_reduced_cost.items() if val <= ub-lb]
        list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val, dict_x = self._solve_scp_resctricted_cluster(list_considered_clusters,relax=False,time_run=1200,warm_start=list_selected_clusters)
        clustering_logger.info('Final interger problem solved, the optimality gap is of '+ str(ub - lb) + ' nb clusters ' + str(len(list_considered_clusters)) + ' final solu ' + str(obj_val) + ' in ' + str(time.time() - time_ub))

        return list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val, dict_x


    def _upper_bound_by_branching(self,dict_x):
        """
        Find an upper bound by the branching rule
        :param dict_x: a dict[clu_id] = x value
        :return: the upper bound found, the list of clusters selected
        """
        time_start = time.time()
        dict_wanted = {600:1,1000:1,1500:2,2000:2,3000:3,5000:5,7500:5,10000:5}
        nb_wanted = dict_wanted[size_data.NUMBER_CUSTOMERS]
        still_to_be_served = list(self.manager_stops.keys()).copy()
        avalailable_cluster = list(self.manager_cluster.keys()).copy()

        final_cluster = []
        comp = 0
        while len(still_to_be_served) >= 0.5:
            list_x_clu_id = [(x,clu_id) for clu_id, x in dict_x.items()]
            list_x_clu_id.sort(reverse=True)
            list_x_clu_id = list_x_clu_id[0:min(nb_wanted,len(list_x_clu_id))]

            list_stop_served = []
            for x,clu_id in list_x_clu_id:
                list_stop_served.extend(list(self.manager_cluster[clu_id].keys()))
                avalailable_cluster.remove(clu_id)
                final_cluster.append(clu_id)

            for stop_id in set(list_stop_served):
                if stop_id in still_to_be_served:
                    still_to_be_served.remove(stop_id)

            dict_stop_cluster = {}
            for stop_id in still_to_be_served:
                dict_stop_cluster[stop_id]= [clu_id for clu_id in self.manager_cluster.dict_stop_clusters[stop_id] if not clu_id in final_cluster]

            scp_mip = scp_solver.MIP_set_covering(list_stop=still_to_be_served,
                                              dict_stop_clus= dict_stop_cluster,
                                              dict_clus_predict= {clu_id : self.manager_cluster[clu_id].expected_prediction for clu_id in avalailable_cluster}) # note the updated prediction with robustness should
                                                                                                                                                    # already have been considered

            list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val, dict_x = scp_mip.solve(relax=True,warm_start=None)
            comp +=1

        ub = sum(self.manager_cluster[clu_id].expected_prediction for clu_id in final_cluster)
        clustering_logger.info('We have done the branching in ' +str(comp) + ' iterations and '+ str(time.time() - time_start)+ 's for an upper bound of '+ str(ub))
        return ub,final_cluster


    def _solve_scp_resctricted_cluster(self,list_clusters,relax,time_run,warm_start = None):
        """
        Solve the scp on a really small amount of clusters
        :param list_clusters: the list of considered clusters
        :param relax: if we want the relax version or not
        :param time_run: set up the time we want it to run
        :param warm_start: a potential warm start, i.e. the list of selected clusters
        :return: list of selected cluster, dict of reduced cost and dual values, objective value, dict of x value
        """
        dict_stop_considered = {}
        for stop_id,list_cluster_serving in self.manager_cluster.dict_stop_clusters.items():
            dict_stop_considered[stop_id] = [clu_id for clu_id in list_cluster_serving if clu_id in list_clusters]

        scp_mip = scp_solver.MIP_set_covering(list_stop=list(self.manager_stops.keys()),
                                              dict_stop_clus= dict_stop_considered,
                                              dict_clus_predict= {clu_id : self.manager_cluster[clu_id].expected_prediction for clu_id in list_clusters}) # note the updated prediction with robustness should
                                                                                                                                                    # already have been considered
        scp_mip.modelOptim.Params.TimeLimit = time_run

        list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val, dict_x  = scp_mip.solve(relax=relax,warm_start=warm_start)

        return list_selected_clusters, dict_reduced_cost, dict_dual_val, obj_val, dict_x


    def _improve_clusters(self,dict_x,dict_reduced_cost, dict_dual_val):
        """
        Improve the current clusters, by selecting the most promising ones (e.g. low reduced cost) and
        trying to either fill them or cleverly remove some stops
        :param dict_x: a dict[cluster_id] = x
        :param dict_reduced_cost: a dict[cluster_id] = reduced cost of the clusters
        :param dict_dual_val: a dict[stop_id] = dual val
        Update the self.manager_clusters
        """
        # re-update the number of clusters tp be improved to speed up the beginning:
        nb_selected = len([clu_id for clu_id,x_val in dict_x.items() if x_val >= 0.9])
        nb_selected = max(nb_selected + 5,self.max_clusters_improved)

        # Sort the clusters by lower reduced cost
        l_cluster_id = [(dict_reduced_cost[clu_id] - dict_x[clu_id],clu_id) for clu_id in dict_x]
        l_cluster_id.sort()
        l_cluster_id = [x[1] for x in l_cluster_id]
        l_cluster_id = l_cluster_id[0:min(nb_selected,len(l_cluster_id))]

        for cluster_to_be_treated in l_cluster_id:
            self.cluster_operation.improve_cluster(cluster_to_be_treated,dict_dual_val,iteration=self.iteration)



    def _merge_clusters(self,dict_x,dict_reduced_cost,dict_dual_val):
        """
        Merge the most interesting clusters by most similarities, for the moment number of stops
        in common. The update the clusters
        :param dict_x: a dict[cluster_id] = x
        :param dict_reduced_cost: a dict[cluster_id] = reduced cost
        :param dict_dual_val: a dict[stop_id] = dual val
        Update the manager clusters
        """
        list_added = self.cluster_merger.merge_clusters(dict_x,dict_reduced_cost,self.iteration)
        for new_cluster_id in list_added:
            self.cluster_operation.improve_cluster(cluster_id=new_cluster_id,
                                                   stop_dual_cost= dict_dual_val,
                                                   iteration=self.iteration)

    def _create_new_clusters(self,dict_reduced_cost,dict_dual_val):
        """
        Identify relevant stops to begin clusters, either based on how many candidates cover them
        or based on their dual value.
        :param dict_reduced_cost: a dict[cluster_id] = reduced cost
        :param dict_dual_val: a dict[stop_id] = dual val
        Update the manager clusters
        """
        list_created = self._create_clusters_based_covertness(dict_reduced_cost, dict_dual_val)
        list_created.extend(self._create_cluster_based_dual_value(dict_dual_val))
        for cluster_id in list_created:
            self.cluster_operation.improve_cluster(cluster_id=cluster_id,
                                                   stop_dual_cost= dict_dual_val,
                                                   iteration=self.iteration)


    def _create_clusters_based_covertness(self, dict_reduced_cost, dict_dual_val):
        """
        Create new clusters based on the current coverntess by nearly optimal clusters
        :param dict_reduced_cost: a dict[cluster_id] = reduced cost
        :param dict_dual_val: a dict[stop_id] = dual val
        Update the manager clusters
        """
        tracker = 'create_covertness'

        # identify the clusters covering it and within the 10 percentile of optimality
        threshold_optim = np.percentile(list(dict_reduced_cost.values()),10)
        list_stop_cluster = []
        for stop_id in self.manager_stops:
            nb_clusters = len([clu_id for clu_id in self.manager_cluster.dict_stop_clusters[stop_id] if dict_reduced_cost[clu_id] <= threshold_optim and self.manager_cluster[clu_id].prediction <= 5])
            list_stop_cluster.append((nb_clusters,dict_dual_val[stop_id],stop_id))  # we add the dual value to differentiate between potential tie.
        list_stop_cluster.sort()
        if self.threshold >= 0.1:
            list_stop_cluster = list_stop_cluster[0:2 * self.number_creation]
        else:
            list_stop_cluster = list_stop_cluster[0:self.number_creation]

        list_created = []
        for nb_clu, dual_val, stop_id in list_stop_cluster:
            new_cluster_id = self.cluster_creation.create_cluster(stop_id,dict_dual_val,(tracker,self.iteration))
            list_created.append(new_cluster_id)
        return list_created


    def _create_cluster_based_dual_value(self, dict_dual_val):
        """
        Create new clusters based on the current dual_value of the stops
        :param dict_dual_val: a dict[stop_id] = dual val
        Update the manager clusters
        """
        tracker = 'create_dual_val'

        list_stop_dual_value = [(val_inv,key) for key,val_inv in dict_dual_val.items()]
        list_stop_dual_value.sort()
        list_stop_dual_value = list_stop_dual_value[0: 2 *self.number_creation]

        list_created = []
        for val_inv,stop_id in list_stop_dual_value:
            new_cluster_id = self.cluster_creation.create_cluster(stop_id,dict_dual_val,(tracker,self.iteration))
            list_created.append(new_cluster_id)
        return list_created


    def check_accuracy(self,list_clu):
        """
        Check the clusters accuracy...
        :return:
        """
        accuracy = 0
        num_total_vehicle = 0
        total_dist = 0
        prediction = 0
        for clus_id in tqdm(list_clu,desc='checking accuracy in main clustering algo'):
            clus = self.manager_cluster[clus_id]
            routing_solver = cvrptw_routing_solver.RoutingSolverCVRPTW(clus,self.config)
            num_vehicle,distance,list_routes = routing_solver.solve_parse_routing()

            total_dist += distance
            num_total_vehicle += num_vehicle

            # check accuracy
            prediction += clus.prediction
            if num_vehicle == clus.prediction:
                clustering_logger.info("accurate " + str(clus.prediction) + ' ' + str(clus.expected_prediction) + " vs " + str(num_vehicle) + " for "+ clus.guid + str(self.manager_cluster[clus_id].is_robust(self.threshold)))
                accuracy += 1
            else:
                clustering_logger.info("error "+ str(clus.prediction) + ' ' + str(clus.expected_prediction) + " vs " + str(num_vehicle) + " for "+ clus.guid + str(self.manager_cluster[clus_id].is_robust(self.threshold)))

        accuracy = accuracy/len(list_clu)
        clustering_logger.info("Number of clusters selected " + str(len(list_clu)))
        clustering_logger.info(" We have an accuracy of " + str(accuracy) + "for prediction" + str(prediction) + " for total number vehu " + str(num_total_vehicle) + " and distance " + str(total_dist))

        return accuracy,num_total_vehicle


    def save_results(self):
        """
        Ouput the convergence results as well as the clusters already created.
        Update everything
        """
        output_cluster = local_path.PATH_TO_PROJECT + 'database/cvrptw/clusters_save.csv'
        data_clusters = self.manager_cluster.output_manager_clusters()
        data_clusters = pd.DataFrame(data_clusters)
        data_clusters.to_csv(output_cluster,header=True,index=False)

        # output stats
        output_file = local_path.PATH_TO_PROJECT + 'database/cvrptw/stats_algo.csv'
        data = []
        row = {'iteration':self.iteration,
               'lp_value':self.lp_value[self.iteration],
               'nb_clusters':self.nb_total_clusters[self.iteration],
               'per_neg_rc':self.per_negative_rc[self.iteration],
               'avg_neg_rc':self.avg_negative_rc[self.iteration],
               'avg_robustness_created': self.avg_robustness_created[self.iteration],
               'predicted_nb_vehi':self.predicted_nb_vehi[self.iteration],
               'sca_accuracy':self.accuracy[self.iteration],
               'real_nb_vehi':self.real_nb_vehi[self.iteration]}
        data.append(row)
        data = pd.DataFrame(data)
        header = True
        if os.path.isfile(output_file):
            header = False
        data.to_csv(output_file,header=header,index=False,mode='a')


    @staticmethod
    def send_email(msg):
        """
        Send an email when the program has either crashed or finished
        :param msg: the message content
        :return:
        """
        # Create a secure SSL context
        context = ssl.create_default_context()

        # set up the SMTP server
        s = smtplib.SMTP_SSL(host="smtp.gmail.com", port=465,context=context)
        s.login(local_path.EMAIL_SENDER_ADDRESS, local_path.EMAIL_PASSWORD)

        email = MIMEMultipart()
        email['From'] = local_path.EMAIL_SENDER_ADDRESS
        email['To'] = local_path.EMAIL_RECEIVER_ADDRESS
        email['Subject'] = msg
        email.attach(MIMEText(msg + local_path.NAME_MACHINE,'plain'))

        s.send_message(email)


if __name__ == '__main__':
    # reset logs
    logging_files.reset_all()

    filename = local_path.PATH_TO_PROJECT + "benchmarks/cvrptw/created_new/1000_customers/test_9.txt"
    size_data.NUMBER_CUSTOMERS = 1000
    man_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    confi = config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)
    print('upload of stops and config done')

    time_init = time.time()
    obj_cre = main_clustering_algo(man_ref,confi)
    obj_cre.create_relevant_clusters(read_cluster=False)
    cluster_selected = obj_cre.solve()
    obj_cre.cluster_operation.print_stats()
    print('Total time spent ', obj_cre.total_time_merge, obj_cre.total_time_improv,obj_cre.total_time_creation)
    print("Total ", time.time() - time_init)
    obj_cre.check_accuracy(list_clu=cluster_selected)

    obj_cre.send_email('Main clustering algo done ')







