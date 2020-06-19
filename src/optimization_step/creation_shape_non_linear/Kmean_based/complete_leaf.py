import copy

from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset
from src.data_set_creation.stops.objects import stops

class CompleteLeaf(object):
    """
    Take a cluster and complete it according to the constraints obtained
    """

    def __init__(self,manager_stop, cluster_stop, dict_leaf_const,initial_leaf_id):
        self.manager_stop = manager_stop
        self.cluster_stop = cluster_stop        # inherits from manager stop, but corresponds to the initial cluster
        self.featurer = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(manager_stop=self.cluster_stop)
        self.current_dict_feature = self._get_feature()

        self.dist_list = self._compute_dist_list()
        self.dict_leaf_const = dict_leaf_const

        # Tree params
        self.current_leaf_id = initial_leaf_id


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


    def complete_greedy_cluster(self,safety_treshold):
        """
        Make sure that the cluster is "full" i.e. that almose all cst are tight
        :param safety_treshold: corresponds to how close from the constraint we want to be
        :return: such a cluster
        """

        # not that we are only interesting in constraint such that <=
        list_cst = [cst for cst in self.dict_leaf_const[self.current_leaf_id] if not cst.is_greater]
        # list_cst = [cst for cst in self.dict_leaf_const[self.current_leaf_id]]

        # since there is no hierarchy in an optimal tree tree each of them once
        self.dist_list.sort()
        list_consider = self.dist_list[0:100]
        assert len(list_consider) == 100, list_consider
        has_done_changes = False
        for dist,stopId in list_consider:
            stop = self.manager_stop[stopId]

            # check if we can add it.
            change = self._check_add_stop(stop,list_cst,safety_treshold)
            if change:
                has_done_changes = True

        if has_done_changes:
            self.cluster_stop.guid= "greedy_" + str(safety_treshold) +"_"+ self.cluster_stop.guid
        return self.cluster_stop

    def _check_add_stop(self,stop,list_cst,safety_threshold):
        """
        Check if we can add the stop to the cluster without having it moving of branch
        :param stop: the stop to be added
        :return: a boolean (and cluster modified if added )
        """
        dict_feature = self._update_feature(stop)

        for cst in list_cst:
            is_satisfied = cst.is_respected(dict_feature,safety_threshold)
            if not is_satisfied:
                assert not stop.guid in self.featurer.manager_stop
                assert stop.guid in self.featurer.distance_matrix
                return False

        self.cluster_stop.add_stop(stop)
        self.current_dict_feature = dict_feature

        # for test
        assert stop.guid in self.featurer.manager_stop
        assert stop.guid in self.featurer.distance_matrix

        return True


    def _get_feature(self):
        """
        :return: a dict[feature name] = value for the considered cluster
        """
        dict_feature = self.featurer.derive_features()

        return dict_feature


    def _update_feature(self,stop):
        """
        Update the dict feature when we add a stop. Note that this may not be exact
        :param stop: the sotp to be added
        :return: a dict feature
        """
        test_dict = copy.deepcopy(self.current_dict_feature)

        return self.featurer.update_feature_add_stop(test_dict,stop,list_feat=list(self.current_dict_feature.keys()))
