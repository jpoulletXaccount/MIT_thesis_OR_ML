
from src.helpers import useful_name,size_data

import numpy as np

class DerivingBestFeatures(object):
    """
    Clsas useful to derive the most relevant features from a manger stops
    """

    def __init__(self,manager_stop):
        self.manager_stop = manager_stop
        self.distance_matrix = manager_stop.matrix_stops_dist
        self.distance_depot = manager_stop.matrix_depot_dist
        self.list_k = [5,10]
        self.TW_BUCKET = 6


    def derive_features(self):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        dict_features.update(self.derive_distance_depot_features(list(self.manager_stop.keys()),guid="_all"))
        dict_features.update(self.derive_distance_stops_features(list(self.manager_stop.keys()),guid="_all"))
        dict_features.update(self.derive_stop_features(list(self.manager_stop.keys()),guid="_all"))
        dict_features.update(self.derive_TW_features())
        dict_features.update(self.derive_overlap_TW_features())

        return dict_features


    def derive_overlap_TW_features(self):
        """
        Derive the overlapping TW features
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.OVERALP + "_number"
        dict_features[key_name] = self.manager_stop.get_avg_number_overlapping_tw_stops(50)

        # mean of k closest stops
        for true_k in self.list_k:

            list_closest = []
            list_max = []

            for stopId in self.manager_stop.keys():
                stop = self.manager_stop[stopId]

                list_overlap = self.manager_stop.get_list_overlapping_stop(stop,threshold=50)

                if len(list_overlap) <= 1:
                    pass

                else:
                    k = min(true_k,len(list_overlap) - 1) - 1
                    list_dist = np.array([self.distance_matrix[stopId][otherId] for otherId in list_overlap if otherId != stopId])
                    idx_closest = np.argpartition(list_dist,k)
                    list_closest.append(np.mean(list_dist[idx_closest[:k+1]]))
                    list_max.append(list_dist[idx_closest[k]])

            # ensures that at list one element
            if len(list_closest) == 0:
                list_closest = [0]
                list_max = [0]

            key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + "_overlap"
            dict_features[key_name] = np.mean(list_closest)
            key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + "_overlap"
            dict_features[key_name] = np.mean(list_max)

        return dict_features


    def derive_TW_features(self):
        """
        derive features by TW bucket
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        nb_class = self.TW_BUCKET
        TW_all = size_data.TW
        inter = TW_all[1] - TW_all[0]
        inter_adv = inter/nb_class

        total_number = 0
        for i in range(0,nb_class):
            begin_TW = TW_all[0] + i *inter_adv
            end_TW = TW_all[0] + (i+1) * inter_adv
            list_stop_id_in = self.manager_stop.get_all_stops_end_in(begin_TW,end_TW)
            total_number += len(list_stop_id_in)
            dict_features.update(self.derive_distance_depot_features(list_stop_id_in,guid=str(end_TW)))
            dict_features.update(self.derive_distance_stops_features(list_stop_id_in,guid=str(end_TW)))
            dict_features.update(self.derive_stop_features(list_stop_id_in,guid = str(end_TW)))

        assert total_number == len(self.manager_stop), str(total_number) + "_" + str(len(self.manager_stop))

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

        # Mean of pairwise distance
        mean_list= np.mean([self.distance_matrix[stopId][otherId] for stopId in manager_considered.keys() for otherId in manager_considered.keys() if otherId != stopId])
        key_name = useful_name.DISTANCE +"_mean_pairwise" + guid
        dict_features[key_name] = mean_list

        # Min of pairwise distance
        min_list= min([self.distance_matrix[stopId][otherId] for stopId in manager_considered.keys() for otherId in manager_considered.keys() if otherId != stopId])
        key_name = useful_name.DISTANCE +"_min_pairwise" + guid
        dict_features[key_name] = np.mean(min_list)

        # mean of k closest stops
        list_dia = []
        for true_k in self.list_k:
            k = min(true_k,len(list_stop_id) - 1) - 1

            list_closest = []
            list_max = []

            for stopId in manager_considered:
                list_dist = np.array([self.distance_matrix[stopId][otherId] for otherId in manager_considered.keys() if otherId != stopId])
                list_dia.append(max(list_dist))
                idx_closest = np.argpartition(list_dist,k)
                list_closest.append(np.mean(list_dist[idx_closest[:k+1]]))
                list_max.append(list_dist[idx_closest[k]])

            key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + guid
            dict_features[key_name] = np.mean(list_closest)
            key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + guid
            dict_features[key_name] = np.mean(list_max)

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

        key_name = useful_name.DIAMETER + guid
        dict_features[key_name] = 0

         # mean of k closest stops
        for true_k in self.list_k:

            key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + guid
            dict_features[key_name] = 0
            key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + guid
            dict_features[key_name] = 0

        return dict_features


    def derive_distance_depot_features(self,list_stop,guid):
        """
        derive all features related to distance between stops and depot
        :param list_stop: the list of stops considered
        :return: a dict['features_name'] = feature value
        """
        if len(list_stop) == 0:
            return self._derive_distance_depot_features_null(guid=guid)

        dict_features = {}
        manager_considered = {stopId : self.manager_stop[stopId] for stopId in list_stop}

        # Mean of pairwise distance
        mean_pairwise= np.mean([self.distance_depot[stopId] for stopId in manager_considered.keys()])
        key_name = useful_name.DEPOT +"_mean_pairwise" + guid
        dict_features[key_name] = mean_pairwise

         # Min of pairwise distance
        min_depot= min([self.distance_depot[stopId] for stopId in manager_considered.keys()])
        key_name = useful_name.DEPOT+"_min_pairwise" + guid
        dict_features[key_name] = min_depot

        # mean of k closest stops
        for true_k in self.list_k:

            k = min(true_k,len(list_stop)) -1

            list_dist = np.array([self.distance_depot[stopId] for stopId in manager_considered.keys()])
            idx_closest = np.argpartition(list_dist,k)
            list_closest = list_dist[idx_closest[:k+1]]
            assert len(list_closest) > 0, list_dist
            list_max = list_dist[idx_closest[k]]

            key_name = useful_name.DEPOT +"_closest_mean_" + str(true_k) + guid
            dict_features[key_name] = np.mean(list_closest)
            key_name = useful_name.DEPOT +"_closest_max" + str(true_k) + guid
            dict_features[key_name] = np.mean(list_max)

        return dict_features


    def _derive_distance_depot_features_null(self,guid):
        """
        Second the derive distance deport features in the case of null list
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        key_name = useful_name.DEPOT +"_mean_pairwise" + guid
        dict_features[key_name] = 0

        key_name = useful_name.DEPOT+"_min_pairwise" + guid
        dict_features[key_name] = 0

        # mean of k closest stops
        for true_k in self.list_k:

            key_name = useful_name.DEPOT +"_closest_mean_" + str(true_k) + guid
            dict_features[key_name] = 0
            key_name = useful_name.DEPOT +"_closest_max" + str(true_k) + guid
            dict_features[key_name] = 0

        return dict_features

