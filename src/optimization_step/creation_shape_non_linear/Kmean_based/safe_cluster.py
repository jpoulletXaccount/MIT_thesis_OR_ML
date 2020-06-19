from src.data_set_creation.stops.objects import stops
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset


class SafeCluster(object):

    def __init__(self,manager_stop, cluster_stop, dict_leaf_const,initial_leaf_id):
        self.manager_stop = manager_stop
        self.cluster_stop = cluster_stop        # inherits from manager stop, but corresponds to the initial cluster
        self.featurer = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(manager_stop=self.cluster_stop)
        self.current_dict_feature = self._get_feature()


        self.dist_list = self._compute_dist_list()
        self.dict_leaf_const = dict_leaf_const
        self.current_leaf_id = initial_leaf_id


    def _compute_dist_list(self):
        """
        For every point not in the cluster, compute its distance to the centroid of the latter
        :return: a list = [(dist,sotp.guid)]
        """
        centroid_cood = self.cluster_stop.centroid
        centroid = stops.Stop("centroid",centroid_cood[0],centroid_cood[1])

        list_dist = []
        for stop in self.cluster_stop.values():
            dist = centroid.get_distance_to_another_stop(stop)
            list_dist.append((dist,stop.guid))

        assert len(list_dist) == len(self.cluster_stop)
        return list_dist


    def _get_feature(self):
        """
        :return: a dict[feature name] = value for the considered cluster
        """
        dict_feature = self.featurer.derive_features()
        return dict_feature

    def _check_need_continue(self,list_cst,safety_threshold):
        """
        CHeck if we need to go on or we can stop there because we are accurate enough
        :param list_cst:
        :return:a boolean and one not respected cst, True if need to continue
        """
        for cst in list_cst:
            if not cst.is_respected(self.current_dict_feature,safety_threshold):
                return True,cst

        return False, -1


    def _improve_cst(self,cst,safety_threshold):
        """
        Improve the cst to make is under the safety threshold
        :param cst:
        :param safety_threshold:
        :return:
        """
        feature_name = cst.get_feature_name(self.current_dict_feature)
        guid = feature_name.split("_")[-1]

        if "all" in guid:
            dist_stop = [(a,b) for a,b in self.dist_list]
        elif "early" in guid:
            dist_stop = [(a,b) for a,b in self.dist_list if self.manager_stop[b].endTW == 250]
        elif "late" in guid:
            dist_stop = [(a,b) for a,b in self.dist_list if self.manager_stop[b].beginTW == 750]
        else:
            assert False, feature_name
        assert len(dist_stop) > 0, cst.to_print()

        dist_stop = [(a,b) for a,b in self.dist_list if b in self.cluster_stop.keys()]
        assert len(dist_stop) == len(self.cluster_stop)

        dist_stop.sort(reverse=True)
        can_stop = False
        comp = 0
        while not can_stop:
            assert comp < len(dist_stop),print(cst.to_print(), self.current_dict_feature)
            assert len(self.cluster_stop) >0, print(cst.to_print(), self.current_dict_feature)
            stop_to_remove = self.manager_stop[dist_stop[comp][1]]
            can_stop= self._check_remove_stop(stop_to_remove,cst,safety_threshold)
            comp +=1


    def _check_remove_stop(self,stop,cst,safety_threshold):
        """
        Check if by removing the stop to the cluster we move further enough from the cst
        :param stop: the stop to be added
        :return: a boolean (and cluster modified if added )
        """
        self.cluster_stop.remove_stop(stop)
        feat_name = cst.get_feature_name(self.current_dict_feature)
        dict_feature = self.featurer.compute_specific_feature(feature_name=feat_name)

        # if we do remove the stops then update
        for keyname in dict_feature:
            assert keyname in self.current_dict_feature
            self.current_dict_feature[keyname] = dict_feature[keyname]

        is_satisfied = cst.is_respected(self.current_dict_feature,safety_threshold)
        if is_satisfied:
            return True
        return False


    def safe_cluster(self,safety_threshold):
        """
        ensure that the cluster is safe, i.e that the constraints are all respected by a certain percentage
        :param safety_threshold: such a percentage
        :return: self.cluter
        """
        list_cst = [cst for cst in self.dict_leaf_const[self.current_leaf_id] if not cst.is_greater]

        need_continue,cst_to_improve = self._check_need_continue(list_cst,safety_threshold)
        if need_continue:
            self.cluster_stop.guid = "safe_" + self.cluster_stop.guid + "_" + str(safety_threshold)
        while need_continue:
            self._improve_cst(cst_to_improve,safety_threshold)
            list_cst.remove(cst_to_improve)
            self.current_dict_feature = self._get_feature()
            need_continue,cst_to_improve = self._check_need_continue(list_cst,safety_threshold)


        return self.cluster_stop




