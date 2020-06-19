from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset
from src.data_set_creation.stops.objects import stops

class SwitchLeaf(object):
    """
    Take a cluster and remove stop to make it go to the lower leaf
    (if possible)
    """

    def __init__(self,manager_stop, cluster_stop, dict_leaf_const,initial_leaf_id):
        self.manager_stop = manager_stop
        self.cluster_stop = cluster_stop        # inherits from manager stop, but corresponds to the initial cluster
        self.featurer = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(manager_stop=self.cluster_stop)

        self.dist_list = self._compute_dist_list()
        self.dict_leaf_const = dict_leaf_const

        # To be updated
        self.current_leaf_id = initial_leaf_id
        self.features = self._get_feature()


    def _get_feature(self):
        """
        :return: a dict[feature name] = value for the considered cluster
        """
        dict_feature = self.featurer.derive_features()

        return dict_feature


    def _compute_dist_list(self):
        """
        For every point not in the cluster, compute its distance to the centroid of the latter
        :return: a list = [(dist,sotp.guid)]
        """
        centroid_cood = self.cluster_stop.centroid
        centroid = stops.Stop("centroid",centroid_cood[0],centroid_cood[1])

        list_dist = []
        for stop in self.manager_stop.values():
            if not stop.guid in self.cluster_stop.keys():
                dist = centroid.get_distance_to_another_stop(stop)
                list_dist.append((dist,stop.guid))

        return list_dist


    def _become_below_cst(self,cst,safety_threshold):
        """
        Make the cluster switch from being greater than the constraints to below
        :param cst: the cst considered
        :param safety_threshold: a safety threshold to be below to.
        :return:
        """
        self._greedy_below(cst,safety_threshold)


    def _greedy_below(self,cst,safety_threshold):
        """
        In a greedy way, remove stop as long as the constraint is still satisfied
        :param cst: the cst we need to go below to
        :param safety_threshold: the safety threshold
        :return:
        """
        feature_name = cst.get_feature_name(self.features)
        guid = feature_name.split("_")[-1]

        if "all" in guid:
            dist_stop = [(self.manager_stop.matrix_depot_dist[stop_id], stop_id) for stop_id in self.cluster_stop.keys()]
        elif "early" in guid:
            dist_stop = [(self.manager_stop.matrix_depot_dist[stop_id], stop_id) for stop_id in self.cluster_stop.keys() if self.cluster_stop[stop_id].endTW == 250]
        elif "late" in guid:
            dist_stop = [(self.manager_stop.matrix_depot_dist[stop_id], stop_id) for stop_id in self.cluster_stop.keys() if self.cluster_stop[stop_id].beginTW == 750]
        else:
            assert False, feature_name
        assert len(dist_stop) > 0, cst.to_print()

        dist_stop.sort(reverse=True)
        can_stop = False
        comp = 0
        while not can_stop:
            stop_to_remove = self.manager_stop[dist_stop[comp][1]]
            can_stop= self._check_remove_stop(stop_to_remove,cst,safety_threshold)
            comp +=1


    def _check_remove_stop(self,stop,cst,safety_threshold):
        """
        Check if by removing the stop to the cluster we move to a new leaf
        :param stop: the stop to be added
        :return: a boolean (and cluster modified if added )
        """
        self.cluster_stop.remove_stop(stop)
        dict_feature = self.featurer.compute_specific_feature(feature_name=cst.get_feature_name(self.features))
        for keyname in dict_feature:
            self.features[keyname] = dict_feature[keyname]
        is_satisfied = cst.is_respected(self.features,-safety_threshold)
        if not is_satisfied:
            return True
        return False


    def decrease_cluster(self,safety_threshold):
        """
        Try to remove some stops so that we diminish one vehicle.
        For simplicity, only concerns about last constraint
        :return: a boolean if can do the change
        """
        last_cst = self.dict_leaf_const[self.current_leaf_id][-1]
        if not last_cst.is_greater:
            return False
        self._become_below_cst(last_cst,safety_threshold)
        self.cluster_stop.guid= "switch_" + str(safety_threshold) +"_"+ self.cluster_stop.guid
        return True


