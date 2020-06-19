
import numpy as np

class CreateClusters(object):
    """
    Class which create clusters based on an initial stop and the current situation of the
    dual value
    """

    def __init__(self,manager_clusters, manager_stops, config, tree, dict_cst):
        self.manager_clusters = manager_clusters
        self.manager_stops = manager_stops
        self.config = config
        self.tree = tree
        self.dict_cst = dict_cst

        # Parameters
        self.weight_score = 0.5
        self.weight_dual_val = 0.5
        self.initial_demand_stops = config.capacity_cst * 1.75


    def create_cluster(self,stop_id,stop_dual_cost,tracking):
        """
        Create a cluster based on the stop_id and the current dual cost values.
        :param stop_id: the guid of the considered intial stop
        :param stop_dual_cost: a dict[stop_id] = dual cost value
        :param tracking: enables us to track how the clsuters have been built
        """
        list_stops_cluster = [stop_id]
        current_demand = self.manager_stops[stop_id].demand

        re_update_score_every = 20
        while current_demand < self.initial_demand_stops and len(list_stops_cluster) < len(self.manager_stops):
            final_score = self._compute_scores(list_stops_cluster,stop_dual_cost)

            if len(final_score) ==0:
                # corresponds to no more stop with a positive dual value
                break

            final_score.sort()
            final_score = final_score[0:min(re_update_score_every,len(final_score))]

            for _, stop_id in final_score:
                list_stops_cluster.append(stop_id)
                current_demand += self.manager_stops[stop_id].demand

        new_id = self.manager_clusters.create_cluster(list_stops_cluster,self.manager_stops,tracking)

        return new_id

    def _compute_scores(self,current_list_cluster, stop_dual_cost):
        """
        For each stop in the manager stop, compute its score of belonging to the cluster
        in creation
        :param current_list_cluster: a list of stop currently inside the cluster
        :param stop_dual_cost: a dict[stop_id] = dual cost
        :return: a list of potential stop to be added with their cost
        """
        final_score = []
        time_dist_scores = self._compute_time_dist_score(current_list_cluster,stop_dual_cost)
        if len(time_dist_scores) == 0:
            return final_score
        percentile_normalizer = np.percentile(list(time_dist_scores.values()),25)
        per_normal_dual = np.percentile([dual for dual in stop_dual_cost.values() if dual > 0],25)

        for stop_id in time_dist_scores:
            score = self.weight_score * time_dist_scores[stop_id]/percentile_normalizer - self.weight_dual_val * stop_dual_cost[stop_id]/per_normal_dual
            final_score.append((score,stop_id))

        return final_score


    def _compute_time_dist_score(self,current_list_cluster,stop_dual_cost):
        """
        Compute a score based on the dist to stops already in the clusters
        :param current_list_cluster: the list of stops_id already in the cluster
        :return: a dict[stop_id] = score
        """
        dict_score = {}
        list_considered = [stop_id for stop_id in self.manager_stops.keys() if stop_dual_cost[stop_id] > 0.01]
        for stop_id in list_considered:
            if not stop_id in current_list_cluster:
                min_dist = min([self.manager_stops.matrix_stops_dist[stop_id][cluster_stop] for cluster_stop in current_list_cluster])

                similar_list= self._identify_similar_stops(stop_id,current_list_cluster,threshold=50)
                if len(similar_list) >0:
                    min_dist_simi = min([self.manager_stops.matrix_stops_dist[stop_id][simi_stop] for simi_stop in similar_list])
                else:
                    min_dist_simi = min_dist

                score = (min_dist + min_dist_simi)/2
                dict_score[stop_id] = score

        return dict_score

    def _identify_similar_stops(self,stop_id,current_list_cluster, threshold):
        """
        Return a list of very similar stops
        :param stop_id: the stop considered
        :param current_list_cluster: a list of current sotps
        :param threshold: float, indicates how close should be
        :return: a list of similar stops id
        """
        list_similar = []
        stop = self.manager_stops[stop_id]
        target_tw = stop.TW

        for stop_id in current_list_cluster:
            stop_tested = self.manager_stops[stop_id]
            tested_TW= stop_tested.TW

            if abs(target_tw[0] - tested_TW[0]) + abs(target_tw[1] - tested_TW[1]) <= threshold:
                list_similar.append(stop_id)

        return list_similar


