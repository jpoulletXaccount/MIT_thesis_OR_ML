

class ClustersMerger(object):
    """
    Class which merges clusters based on heuristics criteria
    """

    def __init__(self, manager_clusters,manager_stops, tree,nb_considered,nb_treated):
        self.manager_cluster = manager_clusters     # Ref to the manager cluster of the main algo
        self.manager_stops = manager_stops
        self.tree = tree

        # Parameters
        self.nb_considered = nb_considered
        self.nb_treated = nb_treated


    def merge_clusters(self, dict_x,dict_reduced_cost,iteration):
        """
        Main function of this class, merge clusters based on how interesting and similar they are
        :param dict_x: a dict[cluster_id] = x
        :param dict_reduced_cost: a dict[cluster_id] = reduced cost
        :param iteration: the iteration at which the merge is performed
        :return: the list of new clusters guid created
        """
        newly_created = []

        # Sort the clusters by lower reduced cost
        l_cluster_id = [(dict_reduced_cost[clu_id] - dict_x[clu_id],clu_id) for clu_id in dict_x]
        l_cluster_id.sort()
        l_cluster_id = [x[1] for x in l_cluster_id]
        l_cluster_id = l_cluster_id[0:min(self.nb_considered, len(l_cluster_id))]

        clusters_seen = []
        nb_treated = 0
        comp = 0
        while nb_treated < self.nb_treated and comp < len(l_cluster_id):
            cluster_to_merge = l_cluster_id[comp]
            comp +=1
            if not cluster_to_merge in clusters_seen:
                merged_cluster_id, new_cluster_id = self._perform_merge(cluster_to_merge, l_cluster_id,clusters_seen,iteration)
                newly_created.append(new_cluster_id)
                clusters_seen.append(cluster_to_merge)
                clusters_seen.append(merged_cluster_id)
                nb_treated += 1

        return newly_created

    def _perform_merge(self, cluster_id, list_cluster_id, list_already_seen,iteration):
        """
        Find the most relevant clusters to merge from the list of cluster id proposed
        :param cluster_id: the cluster we want to merge
        :param list_cluster_id: the full list of potential clusters
        :param list_already_seen: the list of clsuters already dealt with and therefore not useful to try
        :param iteration: the iteration at which the merge is performed
        :return: the guid of the cluster chosen and the guid of the one created by the merge
        """
        max_score = -1
        max_cluster_id = -1
        for other_cluster_id in list_cluster_id:
            if not other_cluster_id in list_already_seen:
                score = self._get_similarity_shared_stops(cluster_id,other_cluster_id)
                if score > max_score:
                    max_score = score
                    max_cluster_id = other_cluster_id

        new_cluster_id = self.manager_cluster.merge_cluster(cluster_id,max_cluster_id,self.manager_stops,iteration)

        return max_cluster_id, new_cluster_id


    def _get_similarity_shared_stops(self, cluster_a_guid, cluster_b_guid):
        """
        Given two clsuters a and b
        :return: the number of common stops
        """
        cluster_a = self.manager_cluster[cluster_a_guid]
        cluster_b = self.manager_cluster[cluster_b_guid]

        common_stops = [stop_id for stop_id in cluster_a.keys() if stop_id in cluster_b.keys()]

        return len(common_stops)



