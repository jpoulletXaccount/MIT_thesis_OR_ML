import random
import time
import numpy as np

from src.optimization_step.scp_approach.heuristics_improvement import Knapsack_add_MIP,least_stops_removal_MIP
from src.helpers.logging_files import clustering_logger

class modify_clusters(object):
    """
    Class which modify clusters, in the sense that they add or remove stops
    """

    def __init__(self, manager_clusters, manager_stops, tree, dict_cst,threshold):
        self.manager_clusters = manager_clusters
        self.manager_stops = manager_stops
        self.tree = tree
        self.dict_cst = dict_cst
        self.robustness_threshold = threshold

        # stats
        self.avg_negative_rc = []
        self.rc_negative_complete = 0
        self.rc_negative_decrease = 0
        self.rc_negative_add_remove = 0
        self.modify_complete = 0
        self.modify_decrease = 0
        self.modify_add_remove = 0
        self.is_robust= []

        self.time_remove_add = 0
        self.time_complete = 0
        self.time_decrease = 0
        self.time_add = 0
        self.time_mip_knapsack = 0
        self.time_mip_least_stops = 0


    @property
    def reduced_cost_negative(self):
        # print('Complete rc ', self.rc_negative_complete, ' decrease rc ', self.rc_negative_decrease, ' add remove ', self.rc_negative_add_remove)
        # print('Average rc ', np.mean(self.avg_negative_rc))
        # print('Modify complete ', self.modify_complete, ' decrease ', self.modify_decrease, ' add remove ', self.modify_add_remove, 'total ', self.total_modify)
        # print('Percentage complete ', self.rc_negative_complete/self.modify_complete, ' decrease ', self.rc_negative_decrease/max(1,self.modify_decrease), ' add remove ', self.rc_negative_add_remove/self.modify_add_remove)
        return self.rc_negative_add_remove + self.rc_negative_decrease + self.rc_negative_complete

    @property
    def total_modify(self):
        return self.modify_complete + self.modify_decrease + self.modify_add_remove


    def improve_cluster(self, cluster_id, stop_dual_cost,iteration):
        """
        Modify the cluster in different way, so as to make it more interesting
        :param cluster_id: the guid of the considered cluster
        :param stop_dual_cost: a dict[stop_id] = dual cost value
        :param iteration: the iteration at which is performed the improvement
        """
        # time_begin = time.time()
        # self.remove_add_stops_cluster(cluster_id,stop_dual_cost,iteration)
        # time_remove = time.time()
        # self.time_remove_add += time_remove-time_begin

        # if self.manager_clusters[cluster_id].prediction >= 1.5:
        #     decrease_cluster_id = self.decrease_leaf_cluster(cluster_id, stop_dual_cost,iteration)
        #
        #     # and then refill it
        #     list_candidate = [stop_id for stop_id, dual in stop_dual_cost.items() if dual >0.01 and not stop_id in self.manager_clusters[decrease_cluster_id].keys()]
        #     if len(list_candidate) >= 0.5:
        #         self.complete_cluster(decrease_cluster_id, stop_dual_cost, list_candidate,iteration)
        #
        #     rc = self._compute_reduced_cost(decrease_cluster_id,stop_dual_cost)
        #     self.modify_decrease +=1
        #     if rc < 0:
        #         self.rc_negative_decrease +=1
        #         self.avg_negative_rc.append(rc)

        list_candidate = self._find_candidates(cluster_id,stop_dual_cost)
        if len(list_candidate) >= 0.5:
            stop_selected,dict_feat_margin,dict_stop_changes = self._compute_feature_addition_stops_selection(cluster_id, list_candidate,stop_dual_cost)

            time_begin = time.time()
            self.remove_add_stops_cluster(cluster_id,stop_dual_cost,iteration,stop_selected,dict_feat_margin,dict_stop_changes)
            time_remove = time.time()
            self.time_remove_add += time_remove-time_begin

            self.add_complete_cluster(cluster_id,stop_selected,stop_dual_cost,iteration)
            time_add = time.time()
            self.time_add += time_add - time_remove

            self.complete_cluster(cluster_id,stop_selected, stop_dual_cost, dict_feat_margin,dict_stop_changes,iteration)
            self.modify_complete +=1
            rc = self._compute_reduced_cost(cluster_id,stop_dual_cost)
            robust = self.manager_clusters[cluster_id].is_robust(self.robustness_threshold)
            self.is_robust.append(robust)
            if rc < 0:
                self.rc_negative_complete+=1
                self.avg_negative_rc.append(rc)

            self.time_complete += time.time() - time_add



    def add_complete_cluster(self,cluster_id,stop_to_add,stop_dual_cost,iteration):
        """
        Add the stop selected to the cluster and then complete it
        :param cluster_id: the baseline cluster
        :param stop_to_add: the list of stops to be added
        :param stop_dual_cost: a dict[stop_id] = dual value
        :param iteration: the iteration at which is performed the improvement
        :return:
        """
        tracker = 'add_complete'

        new_cluster_id = self.manager_clusters.copy_cluster(cluster_id)
        self._brut_addition_stops(new_cluster_id,stop_to_add)

        # complete now
        list_candidate = self._find_candidates(new_cluster_id,stop_dual_cost)
        if len(list_candidate) >= 0.5:
            stop_selected,dict_feat_margin,dict_stop_changes = self._compute_feature_addition_stops_selection(cluster_id=new_cluster_id,list_stop_candidate=list_candidate,stop_dual_cost=stop_dual_cost)
            self.complete_cluster(cluster_id= new_cluster_id, stop_selected= stop_selected,stop_dual_cost=stop_dual_cost,dict_feat_margin=dict_feat_margin,dict_stop_changes=dict_stop_changes,iteration=iteration)
            self.manager_clusters[new_cluster_id].tracking_evolution.pop()

        self.manager_clusters[new_cluster_id].tracking_evolution.append((tracker,iteration))
        self.modify_add_remove +=1
        rc = self._compute_reduced_cost(new_cluster_id,stop_dual_cost)
        robust = self.manager_clusters[new_cluster_id].is_robust(self.robustness_threshold)
        self.is_robust.append(robust)
        if rc < 0:
            self.rc_negative_add_remove +=1
            self.avg_negative_rc.append(rc)


    def remove_add_stops_cluster(self,cluster_id, stop_dual_cost,iteration,stop_selected,dict_feat_margin,dict_stop_changes):
        """
        First remove all/half of the useless stops and then add new stops to complete the cluster
        :param cluster_id: the cluster considered. Note that gonna be coppied
        :param stop_dual_cost: a dict[stop_id] = dual value
        :param iteration: the iteration at which is performed the improvement
        :param stop_selected: a list of stop to try to add
        :param dict_feat_margin: a dict of slack/surplus for the active cst
        :param dict_stop_changes: the linear changes to the feat if the stop is added
        :return:
        """
        tracker = 'remove_add_'

        list_number_stop = [10,20]
        remove_all = [stop_id for stop_id in self.manager_clusters[cluster_id] if stop_dual_cost[stop_id]<0.01]
        list_number_stop = [i for i in list_number_stop if i <= len(remove_all)-10]

        for nb_stop in list_number_stop:
            current_tracker = tracker + str(nb_stop)
            new_cluster_id_partial = self.manager_clusters.copy_cluster(cluster_id)
            random.shuffle(remove_all)
            remove_partial = remove_all[0:nb_stop]

            self._brut_remove_stops(new_cluster_id_partial,remove_partial)

            # list_candidate = self._find_candidates(new_cluster_id_partial,stop_dual_cost)
            # if len(list_candidate) >= 0.5:
            #     stop_selected,dict_feat_margin,dict_stop_changes = self._compute_feature_addition_stops_selection(cluster_id=new_cluster_id_partial,list_stop_candidate=list_candidate,stop_dual_cost=stop_dual_cost)
            #     self.complete_cluster(cluster_id= new_cluster_id_partial, stop_selected= stop_selected,stop_dual_cost=stop_dual_cost,dict_feat_margin=dict_feat_margin,dict_stop_changes=dict_stop_changes,iteration=iteration)
            #     self.manager_clusters[new_cluster_id_partial].tracking_evolution.pop()

            self.complete_cluster(cluster_id= new_cluster_id_partial, stop_selected= stop_selected,stop_dual_cost=stop_dual_cost,dict_feat_margin=dict_feat_margin,dict_stop_changes=dict_stop_changes,iteration=iteration)
            self.manager_clusters[new_cluster_id_partial].tracking_evolution.pop()

            self.manager_clusters[new_cluster_id_partial].tracking_evolution.append((current_tracker,iteration))
            self.modify_add_remove +=1
            rc = self._compute_reduced_cost(new_cluster_id_partial,stop_dual_cost)
            robust = self.manager_clusters[new_cluster_id_partial].is_robust(self.robustness_threshold)
            self.is_robust.append(robust)
            if rc < 0:
                self.rc_negative_add_remove +=1
                self.avg_negative_rc.append(rc)


    def complete_cluster(self, cluster_id, stop_selected,stop_dual_cost,dict_feat_margin,dict_stop_changes ,iteration):
        """
        Add as many stops as possible from the list of stops candidate, by making sure we stay in the
        same tree leaf
        :param cluster_id: the guid of the considered cluster
        :param stop_dual_cost: a dict[stop_id] = dual value
        :param stop_selected: a list of stop to try to add
        :param dict_feat_margin: a dict of slack/surplus for the active cst
        :param dict_stop_changes: the linear changes to the feat if the stop is added
        :param iteration: the iteration at which is performed the improvement
        """
        self._insert_as_many_stops(cluster_id,stop_selected,dict_feat_margin,dict_stop_changes,stop_dual_cost)
        tracker = 'complete_leaf'
        self.manager_clusters[cluster_id].tracking_evolution.append((tracker,iteration))


    def _compute_feature_addition_stops_selection(self, cluster_id, list_stop_candidate,stop_dual_cost):
        """
        Compute the changes of adding stops and select the most performing one
        :return: the list of selected stops
        """
        dict_stop_changes = self._get_features_addition(cluster_id,list_stop_candidate)
        dict_feat_margin = self._get_features_cap(cluster_id)
        dict_stop_changes = self._complete_feat_weigh(dict_feat_margin,dict_stop_changes)
        time_begin = time.time()
        knapscak_solver = Knapsack_add_MIP.KnapsackFeaturesMIP(dict_feat_cap=dict_feat_margin,dict_stop_feat_weight=dict_stop_changes,
                                                               dict_stop_dual=stop_dual_cost)
        stop_selected = knapscak_solver.solve()
        self.time_mip_knapsack += time.time() - time_begin

        return stop_selected,dict_feat_margin,dict_stop_changes


    @staticmethod
    def _complete_feat_weigh(dict_feat_cap,dict_stop_feat_weigh):
        """
        Complete the weight of stop per featuer by zero.
        :param dict_feat_cap: a dict[feat] = margin
        :param dict_stop_feat_weigh: a dict[stop_id][feat] = weight
        :return: a new dict_stop_feat_weight
        """
        list_feat = list(dict_feat_cap.keys())
        for stop_id in dict_stop_feat_weigh:
            for feat in list_feat:
                if not feat in dict_stop_feat_weigh[stop_id]:
                    dict_stop_feat_weigh[stop_id][feat] = 0

        return dict_stop_feat_weigh


    def _get_features_cap(self,cluster_id):
        """
        Go through all the active cosntraint for the cluster and return the margin
        per features
        :param cluster_id: the considered cluster
        :return: a dict[featuere]= margin
        """
        cluster = self.manager_clusters[cluster_id]
        list_cst = self.dict_cst[str(cluster.leaf)]
        dict_margin_features = {}
        for cst in list_cst:
            feat_name = cst.get_feature_name(cluster.dict_features)
            true_threshold = self.robustness_threshold * self.manager_clusters.dict_dispersion[feat_name]
            dict_margin_features.update(cst.get_margin(cluster.dict_features,threshold=true_threshold))

        return dict_margin_features

    def _get_features_addition(self,cluster_id,list_stop_candidate):
        """
        Get the changes in features if we add the stops.
        :param cluster_id: the guid of the considered cluster
        :param list_stop_candidate: a list of potential stop to be added
        :return a dict[stop_id][feature_name] = changes
        """
        cluster = self.manager_clusters[cluster_id]
        list_cst = self.dict_cst[str(cluster.leaf)]
        list_feat = [cst.get_feature_name(cluster.dict_features) for cst in list_cst]
        dict_stop_addition = {}
        for stop_id in list_stop_candidate:
            stop = self.manager_stops[stop_id]
            update_dict = cluster.featurer.update_feature_add_stop(cluster.dict_features,stop,list_feat)
            dict_changes = {}
            for key_name in update_dict:
                dict_changes[key_name] = update_dict[key_name] - cluster.dict_features[key_name]

            dict_stop_addition[stop_id] = dict_changes

        return dict_stop_addition


    def _get_features_removal(self,cluster_id,list_stop_candidate):
        """
        Get the changes in features if we remove the stops.
        :param cluster_id: the guid of the considered cluster
        :param list_stop_candidate: a list of potential stop to be added
        :return a dict[stop_id][feature_name] = changes
        """
        cluster = self.manager_clusters[cluster_id]
        list_cst = self.dict_cst[str(cluster.leaf)]
        list_feat = [cst.get_feature_name(cluster.dict_features) for cst in list_cst]
        dict_stop_removal = {}
        for stop_id in list_stop_candidate:
            stop = self.manager_stops[stop_id]
            update_dict = cluster.featurer.update_feature_remove_stop(cluster.dict_features,stop,list_feat)
            dict_changes = {}
            for key_name in update_dict:
                dict_changes[key_name] = cluster.dict_features[key_name] - update_dict[key_name]

            dict_stop_removal[stop_id] = dict_changes

        return dict_stop_removal


    def _insert_as_many_stops(self,cluster_id,stops_selected,dict_feat_margin,dict_stop_changes,stop_dual_cost):
        """
        Among the stops selected, try to insert as many of them as possible. To do so, sort them by the
        ration of dual value on weight to the constraints.
        :param cluster_id: the cluster in which we should insert them.
        :param stops_selected: a list of stops selected by hte MIP
        :param dict_feat_margin: a dict[feature] = margin
        :param dict_stop_changes: a dict[stop_id][feature] = impact
        :param stop_dual_cost: a dict[stop_id] = dual cost
        :return:
        """
        # Sort the stops per score
        list_score_stop = []
        for stop_id in stops_selected:
            impact_feature = sum(abs(dict_stop_changes[stop_id][feat]/max(0.001,abs(dict_feat_margin[feat]))) for feat in dict_feat_margin.keys())
            impact_feature = max(impact_feature,0.001)
            list_score_stop.append((stop_dual_cost[stop_id]/impact_feature,stop_id))
        list_score_stop.sort(reverse=True)

        # Try to add the stops
        cluster = self.manager_clusters[cluster_id]
        previous_leaf = cluster.leaf
        list_cst = self.dict_cst[str(cluster.leaf)]
        list_feat = [cst.get_feature_name(cluster.dict_features) for cst in list_cst]
        update_dict = {}
        feas = True
        for score, stop_id in list_score_stop:
            update_dict = cluster.featurer.update_feature_add_stop(cluster.dict_features,self.manager_stops[stop_id],list_feat)
            test_dict= {feat:val for feat,val in cluster.dict_features.items()}
            test_dict.update(update_dict)
            # check if feasible:
            feas = True
            for cst in list_cst:
                feat_name = cst.get_feature_name(test_dict)
                true_threshold = self.manager_clusters.dict_dispersion[feat_name] * self.robustness_threshold
                if not cst.is_respected(test_dict,threshold=true_threshold):
                    feas = False
                    break
            if feas:
                self.manager_clusters.add_stop_to_cluster(self.manager_stops[stop_id],cluster_id)
                cluster.update_features(update_dict,should_be_respected=True)

        # Reclean the dictfeatures of the cluster and check same leaf
        cluster.reset_features(already_updated=list(update_dict.keys()))
        # quick check:
        if feas:
            for feat in update_dict:
                assert abs(update_dict[feat] - cluster.dict_features[feat]) <=0.001, print(feat,update_dict[feat], cluster.dict_features[feat])

        assert cluster.leaf == previous_leaf, print(cluster.leaf,previous_leaf)


    def _brut_remove_stops(self,cluster_id,stops_selected):
        """
        Remove from the cluster all the stops
        :param cluster_id: the id of the clusters
        :param stops_selected: the stops selected
        :return:
        """
        for stop_id in stops_selected:
            self.manager_clusters.remove_stop_from_cluster(self.manager_stops[stop_id],cluster_id)

        # update the features
        self.manager_clusters[cluster_id].reset_features(already_updated=[])


    def _brut_addition_stops(self,cluster_id,stops_selected):
        """
        Add to the cluster all the stops
        :param cluster_id: the id of the clusters
        :param stops_selected: the stops selected
        :return:
        """
        for stop_id in stops_selected:
            self.manager_clusters.add_stop_to_cluster(self.manager_stops[stop_id],cluster_id)

        # update the features
        self.manager_clusters[cluster_id].reset_features(already_updated=[])



    def _remove_stops(self,cluster_id,stops_selected,dict_feat_margin,dict_stop_changes,stop_dual_cost,least):
        """
        Remove all stops from the cluster.
        :param cluster_id: the cluster from which we should remove them.
        :param stops_selected: a list of stops selected by the MIP
        :param dict_feat_margin: a dict[feature] = margin
        :param dict_stop_changes: a dict[stop_id][feature] = impact
        :param stop_dual_cost: a dict[stop_id] = dual cost
        :param least: a boolean indicating if we stop as soon as we have changed leaf or not.
        :return:
        """
        # Sort the stops per score
        list_score_stop = []
        for stop_id in stops_selected:
            impact_feature = sum(abs(dict_stop_changes[stop_id][feat]/max(0.001,dict_feat_margin[feat])) for feat in dict_feat_margin.keys())
            list_score_stop.append((impact_feature/(1 + 10 * stop_dual_cost[stop_id]),stop_id))
        list_score_stop.sort(reverse=True)

        successful = False

        # Try to remove the stops
        cluster = self.manager_clusters[cluster_id]
        previous_leaf = cluster.leaf
        list_cst = self.dict_cst[str(cluster.leaf)]
        list_feat = [cst.get_feature_name(cluster.dict_features) for cst in list_cst]
        update_dict = {}
        for score, stop_id in list_score_stop:
            update_dict = cluster.featurer.update_feature_remove_stop(cluster.dict_features,self.manager_stops[stop_id],list_feat)
            test_dict= {feat:val for feat,val in cluster.dict_features.items()}
            test_dict.update(update_dict)

            # check if becomes violated
            for cst in list_cst:
                feat_name = cst.get_feature_name(test_dict)
                true_threshold = self.manager_clusters.dict_dispersion[feat_name] * self.robustness_threshold
                if not cst.is_respected(test_dict,threshold= -true_threshold):
                    successful = True
                    break
            else:
                successful = False

            self.manager_clusters.remove_stop_from_cluster(self.manager_stops[stop_id],cluster_id)
            cluster.update_features(update_dict,should_be_respected=not successful)

            if successful and least:
                break

        # Reclean the dictfeatures of the cluster and check same leaf
        cluster.reset_features(already_updated=list(update_dict.keys()))
        # quick check:
        for feat in update_dict:
            assert abs(update_dict[feat] - cluster.dict_features[feat]) <=0.001, print(feat,update_dict[feat], cluster.dict_features[feat])

        if successful:
            assert cluster.leaf != previous_leaf, print(cluster.leaf,previous_leaf)


    def decrease_leaf_cluster(self, cluster_id, stop_dual_cost,iteration):
        """
        Remove as little stops as required to make the cluster switch (decrease) leaf
        :param cluster_id: the guid of the considered cluster
        :param stop_dual_cost: a dict[stop_id] = dual value
        :param iteration: the iteration at which is performed the improvement
        :return the new cluster_id
        """
        tracker = 'decrease_leaf'

        initial_cluster = self.manager_clusters[cluster_id]
        previous_leaf = initial_cluster.leaf
        new_cluster_id = self.manager_clusters.copy_cluster(cluster_id)
        new_cluster = self.manager_clusters[new_cluster_id]

        number_stops_to_consider = 30

        while new_cluster.leaf == previous_leaf and len(new_cluster) > 0:
            list_stop_less_impact_reduced_cost = list(new_cluster.keys())
            list_stop_less_impact_reduced_cost.sort(key=lambda x: stop_dual_cost[x])
            list_stop_less_impact_reduced_cost = list_stop_less_impact_reduced_cost[0:min(number_stops_to_consider,len(new_cluster.keys()))]

            # try to remove those
            dict_stop_changes = self._get_features_removal(new_cluster_id,list_stop_less_impact_reduced_cost)
            dict_feat_margin = self._get_features_cap(new_cluster_id)
            dict_feat_margin = {k: -val for k,val in dict_feat_margin.items()}  # We have to reverse the cap to obtain the gap
            dict_stop_changes = self._complete_feat_weigh(dict_feat_margin,dict_stop_changes)
            time_begin = time.time()
            mip_removal_solver = least_stops_removal_MIP.LeastStopsRemovalMIP(dict_feat_gap=dict_feat_margin,dict_stop_feat_weight=dict_stop_changes,
                                                                   dict_stop_dual_val=stop_dual_cost)
            stop_selected = mip_removal_solver.solve()
            self.time_mip_least_stops += time.time() - time_begin
            assert len(stop_selected) > 0, print(new_cluster.leaf, dict_stop_changes)
            self._remove_stops(new_cluster_id,stop_selected,dict_feat_margin,dict_stop_changes,stop_dual_cost,least=True)

        assert new_cluster.leaf != previous_leaf or len(new_cluster) == 0

        new_cluster.tracking_evolution.append((tracker,iteration))

        return new_cluster_id


    def _find_candidates(self,cluster_id,stop_dual_cost):
        """
        Find the most promising candidates for the considered cluster. Based on the dual cost but also making sure that they are not too far away
        :param cluster_id: the cluster considered
        :param stop_dual_cost: the dict[stop_id] = dual value
        :return: a list of stop id
        """
        max_number_targeted = 45

        threshold = np.percentile(list(stop_dual_cost.values()),75)
        threshold = max(threshold,0.01)
        cluster = self.manager_clusters[cluster_id]
        cluster_stop = list(cluster.keys())
        list_candidate = [stop_id for stop_id, dual in stop_dual_cost.items() if dual >threshold and not stop_id in cluster_stop]

        if len(list_candidate) >= max_number_targeted:
            list_candidate = [(stop_dual_cost[stop_id],stop_id) for stop_id in list_candidate if np.linalg.norm(np.array(self.manager_stops[stop_id].xy) - np.array(cluster.centroid)) <= 150]
            list_candidate.sort(reverse=True)
            list_candidate = list_candidate[0:min(max_number_targeted,len(list_candidate))]
            list_candidate = [b for a,b in list_candidate]

        return list_candidate


    def _compute_reduced_cost(self,cluster_id,dict_dual_value):
        """
        Compute the reduced cost of a given cluster
        :return: the reduced cost of the cluster
        """
        cost_cluster = self.manager_clusters[cluster_id].expected_prediction
        sum_lambda = sum(dict_dual_value[stop_id] for stop_id in self.manager_clusters[cluster_id])
        final_reduced_cost = cost_cluster - sum_lambda
        return final_reduced_cost


    def reset_stats(self):
        """
        Reset all the stats of the class
        """
        self.print_stats()
        self.rc_negative_complete = 0
        self.rc_negative_decrease = 0
        self.rc_negative_add_remove = 0
        self.time_mip_least_stops = 0
        self.time_mip_knapsack = 0
        self.time_remove_add = 0
        self.time_complete = 0
        self.time_decrease = 0
        self.time_add =0
        self.modify_complete = 0
        self.modify_decrease = 0
        self.modify_add_remove = 0
        self.avg_negative_rc = []
        self.is_robust = []


    def print_stats(self):
        """
        Print a few stats corresponding to time
        """
        clustering_logger.info('Time spent in remove add ' + str(self.time_remove_add) + ' and decrease '+ str(self.time_decrease) + ' and complete '+ str(self.time_complete) + ' and add then complete '+ str(self.time_add))
        clustering_logger.info('Time spent in knapsack mip '+ str(self.time_mip_knapsack) + ' and in mip least removal ' + str(self.time_mip_least_stops))

