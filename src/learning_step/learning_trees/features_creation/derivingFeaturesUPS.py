import numpy as np

from src.learning_step.learning_trees.update_features import updater_add_create_dataset
from src.helpers import useful_name, size_data

class DerivingFeaturesUPS(object):
    """
    Class which derives features relevant to the created dataset
    """

    def __init__(self,manager_stop, project=False):
        self.manager_stop = manager_stop
        self.distance_matrix = manager_stop.matrix_stops_dist
        self.distance_depot = manager_stop.matrix_depot_dist
        self.project = project

        # Parameter
        self.SIZE_BALL = 10
        self.PERCENTAGE_DENSE = 0.90        # of the average density
        self.PERCENTAGE_SPARSE = 0.50       # of the average density

        self.list_k = [1,2,10]
        self.list_percentile = [25,50,75,100]


    def derive_features(self):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        dict_features.update(self.derive_density_feature(list(self.manager_stop.keys()),guid="_all"))
        dict_features.update(self.derive_depot_features(list(self.manager_stop.keys()),guid="_all"))
        dict_features.update(self.derive_heuristic_feature(list(self.manager_stop.keys()),guid= "_all"))
        dict_features.update(self.derive_stop_features(list(self.manager_stop.keys()),guid="_all"))
        dict_features.update(self.derive_distance_stops_features(list(self.manager_stop.keys()),guid='_all'))
        dict_features.update(self.derive_thight_tw(list(self.manager_stop.keys()),guid='_all'))
        dict_features.update(self.derive_TW_features())

        return dict_features

    def derive_thight_tw(self,list_stop_id,guid):
        """
        Derive features on tight tw
        :return: a dict['features_name'] = feature value
        """
        stop_very_tight = [stop_id for stop_id in list_stop_id if self.manager_stop[stop_id].is_tight(50)]
        stop_tight = [stop_id for stop_id in list_stop_id if self.manager_stop[stop_id].is_tight(100) and not stop_id in stop_very_tight]

        dict_features = {}

        key_name = useful_name.NB_STOPS + '_very_tight_' + guid
        dict_features[key_name] = len(stop_very_tight)

        key_name = useful_name.DEMAND + '_very_tight_' + guid
        dict_features[key_name] = sum(self.manager_stop[stop_id].demand for stop_id in stop_very_tight)

        key_name = useful_name.NB_STOPS + '_tight_' + guid
        dict_features[key_name] = len(stop_tight)

        key_name = useful_name.DEMAND + '_tight_' + guid
        dict_features[key_name] = sum(self.manager_stop[stop_id].demand for stop_id in stop_tight)

        return dict_features

    def derive_stop_features(self,list_stop_id,guid):
        """
        Derive all features corresponding to stops caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.NB_STOPS + guid
        dict_features[key_name] = len(list_stop_id)

        key_name = useful_name.DEMAND + guid
        dict_features[key_name] = sum(self.manager_stop[stopId].demand for stopId in list_stop_id)

        return dict_features

    def derive_distance_stops_features(self,list_stop_id,guid):
        """
        derive all features related to distance between stops
        :return: a dict['features_name'] = feature value
        """
        if len(list_stop_id) <=1:
            return self._derive_distance_stops_features(guid)

        dict_features = {}
        manager_considered = {stopId : self.manager_stop[stopId] for stopId in list_stop_id}

        pairwise_dist = [self.distance_matrix[stopId][otherId] for stopId in manager_considered.keys() for otherId in manager_considered.keys() if otherId != stopId]
        # Mean of pairwise distance
        mean_list= np.mean(pairwise_dist)
        key_name = useful_name.DISTANCE +"_mean_pairwise" + guid
        dict_features[key_name] = mean_list

        # Min of pairwise distance
        min_list= min(pairwise_dist)
        key_name = useful_name.DISTANCE +"_min_pairwise" + guid
        dict_features[key_name] = min_list

        # Max of pairwise distance
        max_list = max(pairwise_dist)
        key_name = useful_name.DISTANCE +"_max_pairwise" + guid
        dict_features[key_name] = max_list

        # mean of k closest stops
        list_dia = []
        for true_k in self.list_k:
            k = min(true_k,len(list_stop_id) - 1) - 1

            list_closest = []
            list_max = []

            for stopId in manager_considered:
                serv_time = manager_considered[stopId].service_time
                list_dist = np.array([self.distance_matrix[stopId][otherId] + serv_time for otherId in manager_considered.keys() if otherId != stopId])
                list_dia.append(max(list_dist))
                idx_closest = np.argpartition(list_dist,k)
                list_closest.append(np.mean(list_dist[idx_closest[:k+1]]))
                list_max.append(list_dist[idx_closest[k]])

            for percent in self.list_percentile:
                key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] =np.percentile(list_closest,percent)
                key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] = np.percentile(list_max,percent)

        key_name = useful_name.DIAMETER + guid
        dict_features[key_name] = max(list_dia)

        return dict_features

    def _derive_distance_stops_features(self,guid):
        """
        Second the derive distance stops features in the case of null list
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        key_name = useful_name.DISTANCE +"_mean_pairwise" + guid
        dict_features[key_name] = 0

        key_name = useful_name.DISTANCE +"_min_pairwise" + guid
        dict_features[key_name] = 0

        key_name = useful_name.DISTANCE +"_max_pairwise" + guid
        dict_features[key_name] = 0

        key_name = useful_name.DIAMETER + guid
        dict_features[key_name] = 0

         # mean of k closest stops
        for true_k in self.list_k:
            for percent in self.list_percentile:
                key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] = 0
                key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] = 0

        return dict_features

    def derive_density_feature(self,list_stop_id,guid):
        """
        Derive the density feature
        :param list_stop_id: the list of stop to be considered
        :param guid: the guid of this feature
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        if len(list_stop_id) ==0:
            key_name = "not_" +useful_name.SPARSE + guid
            dict_features[key_name] = 1

            key_name = useful_name.DENSE + guid
            dict_features[key_name] = 0

            key_name = useful_name.DIAMETER + guid
            dict_features[key_name] = 0

            return dict_features

        size = size_data.SIZE_HOMBERGER
        total_area = (size[0][1] - size[0][0]) * (size[1][1] - size[1][0])
        if guid != "_all":
            threshold_sparse = 0.3 * self.PERCENTAGE_SPARSE * size_data.NUMBER_CUSTOMERS * np.pi * (self.SIZE_BALL**2) / total_area
            threshold_dense = 0.3 * self.PERCENTAGE_DENSE * size_data.NUMBER_CUSTOMERS * np.pi * (self.SIZE_BALL**2) / total_area
        else:
            threshold_sparse =  self.PERCENTAGE_SPARSE * size_data.NUMBER_CUSTOMERS * np.pi * (self.SIZE_BALL**2) / total_area
            threshold_dense = self.PERCENTAGE_DENSE * size_data.NUMBER_CUSTOMERS * np.pi * (self.SIZE_BALL**2) / total_area

        nb_sparse = 0
        nb_dense = 0
        diameter = 0

        for stop_id_1 in list_stop_id:
            list_dist = [self.distance_matrix[stop_id_1][stop_id_2] for stop_id_2 in list_stop_id]
            diameter = max(diameter,max(list_dist))
            nb_close = np.count_nonzero(np.array(list_dist) <= self.SIZE_BALL)
            if nb_close >= threshold_dense:
                nb_dense +=1
            elif nb_close <= threshold_sparse:
                nb_sparse +=1

        key_name = "not_" + useful_name.SPARSE + guid
        dict_features[key_name] = len(list_stop_id) - nb_sparse

        key_name = useful_name.DENSE + guid
        dict_features[key_name] = nb_dense

        key_name = useful_name.DIAMETER + guid
        dict_features[key_name] = diameter

        return dict_features


    def derive_depot_features(self,list_stop,guid):
        """
        derive all features related to distance between stops and depot
        :param list_stop: the list of stops considered
        :param guid: the guid associated to this feature
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        if len(list_stop) ==0:
            key_name = useful_name.DEPOT +"_mean_pairwise" + guid
            dict_features[key_name] = 0
            key_name = useful_name.DEPOT+"_min_pairwise" + guid
            dict_features[key_name] = 0
            key_name = useful_name.DEPOT+"_max_pairwise" + guid
            dict_features[key_name] = 0
            return dict_features

        dist_depot = [self.distance_depot[stopId] for stopId in list_stop]

        # Min of pairwise distance
        min_pairwise= min(dist_depot)
        key_name = useful_name.DEPOT +"_min_pairwise" + guid
        dict_features[key_name] = min_pairwise

        # Mean of pairwise distance
        mean_pairwise= np.mean(dist_depot)
        key_name = useful_name.DEPOT +"_mean_pairwise" + guid
        dict_features[key_name] = mean_pairwise

        # Max of pairwise distance
        max_depot= max(dist_depot)
        key_name = useful_name.DEPOT+"_max_pairwise" + guid
        dict_features[key_name] = max_depot
        return dict_features


    def derive_heuristic_feature(self,list_stop,guid):
        """
        derive all features related to tour of stops
        :param list_stop: the list of stops considered
        :param guid: the guid associated to this feature
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.TSP + guid
        if len(list_stop) == 0:
            dict_features[key_name] = 0
            return dict_features


        list_to_treat = list_stop.copy()

        # add depot
        tot_dist = 0
        dist_depot = [(self.distance_depot[stop_id], stop_id) for stop_id in list_to_treat]
        dist,previous_stop = min(dist_depot)
        tot_dist += dist
        list_to_treat.remove(previous_stop)

        while len(list_to_treat) >=1:
            dist_considered = [(self.distance_matrix[previous_stop][stop_id],stop_id) for stop_id in list_to_treat]
            dist,previous_stop = min(dist_considered)
            tot_dist += dist
            list_to_treat.remove(previous_stop)

        # add depot
        tot_dist += self.distance_depot[previous_stop]

        dict_features[key_name] = tot_dist
        return dict_features


    def derive_TW_features(self):
        """
        derive features by TW bucket
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        list_time_before = [950,1000, 1050, 1400, 1600]
        list_time_after = [1450,1650,1750,1800]
        if not self.project:
            list_time_before = [l-892 for l in list_time_before]
        list_time_before.sort()
        if not self.project:
            list_time_after = [l-892 for l in list_time_after]
        list_time_after.sort(reverse=True)

        has_seen_after = set()
        dict_after = {l : [] for l in list_time_after}
        has_seen_before = set()
        dict_before = {l: [] for l in list_time_before}
        for stop in self.manager_stop.values():
            btw,etw = stop.TW
            for l_before in list_time_before:
                if not stop.guid in has_seen_before:
                    if etw <= l_before:
                        has_seen_before.add(stop.guid)
                        dict_before[l_before].append(stop.guid)

            for l_after in list_time_after:
                if not stop.guid in has_seen_after:
                    if btw >= l_after:
                        has_seen_after.add(stop.guid)
                        dict_after[l_after].append(stop.guid)

        for l_before, stop_before in dict_before.items():
            feature_dict = "before_" + str(l_before)
            dict_features.update(self.derive_depot_features(stop_before,guid=feature_dict))
            dict_features.update(self.derive_stop_features(stop_before,guid=feature_dict))
            dict_features.update(self.derive_density_feature(stop_before,guid = feature_dict))
            dict_features.update(self.derive_heuristic_feature(stop_before,guid = feature_dict))
            dict_features.update(self.derive_distance_stops_features(stop_before,guid = feature_dict))

        for l_after, stop_after in dict_after.items():
            feature_dict = "after_" + str(l_after)
            dict_features.update(self.derive_depot_features(stop_after,guid=feature_dict))
            dict_features.update(self.derive_stop_features(stop_after,guid=feature_dict))
            dict_features.update(self.derive_density_feature(stop_after,guid = feature_dict))
            dict_features.update(self.derive_heuristic_feature(stop_after,guid = feature_dict))
            dict_features.update(self.derive_distance_stops_features(stop_after,guid = feature_dict))

        return dict_features


    def update_feature_add_stop(self,dict_feature, stop,list_feat):
        """
        Update the dict feature if we add the stop
        :param dict_feature: the previous dict_feature
        :param stop: the stop to be added
        :param list_feat: the list of features we want to update
        :return: a dict feature
        """
        update = updater_add_create_dataset.UpdaterAddFeatureCreateDataSet(manager_stop=self.manager_stop,
                                                                    distance_matrix=self.distance_matrix,
                                                                    distance_depot=self.distance_depot,
                                                                    size_ball=self.SIZE_BALL,
                                                                    per_dense=self.PERCENTAGE_DENSE,
                                                                    per_sparse=self.PERCENTAGE_SPARSE,
                                                                    list_k=self.list_k,
                                                                    list_percentile=self.list_percentile,
                                                                    stop_added=stop)
        new_dict_featuer = update.compute_feature_add_stop(dict_feature.copy(),list_feat)
        self.distance_matrix = update.distance_matrix
        self.distance_depot = update.distance_depot
        return new_dict_featuer


    def compute_specific_feature(self,feature_name):
        """
        :param feature_name: the name of the featuer to be computed
        :return: the value
        """
        guid = feature_name.split("_")[-1]

        list_considered = []
        if "all" in guid:
            list_considered = list(self.manager_stop.keys())
        elif "early" in guid:
            list_bucket_early = []
            for stop in self.manager_stop.values():
                btw,etw = stop.TW
                if etw == 250:
                    list_bucket_early.append(stop.guid)
            list_considered = list_bucket_early

        elif "late" in guid:
            list_bucket_late = []

            for stop in self.manager_stop.values():
                btw,etw = stop.TW
                if btw == 750:
                    list_bucket_late.append(stop.guid)
                list_considered = list_bucket_late
        else:
            print(feature_name)
            assert False

        if useful_name.DEMAND in feature_name or useful_name.NB_STOPS in feature_name:
            return self.derive_stop_features(list_stop_id=list_considered,guid="_" +guid)
        elif useful_name.SPARSE in feature_name or useful_name.DENSE in feature_name or useful_name.DIAMETER in feature_name:
            return self.derive_density_feature(list_considered,"_" +guid)

        elif useful_name.DEPOT in feature_name:
            return self.derive_depot_features(list_considered,"_" +guid)

        elif useful_name.TSP in feature_name:
            return self.derive_heuristic_feature(list_considered,"_" +guid)

        else:
            print(feature_name)
            assert False





