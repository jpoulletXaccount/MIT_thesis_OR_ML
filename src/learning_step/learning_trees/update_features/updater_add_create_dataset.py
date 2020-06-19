import numpy as np
import scipy

from src.helpers import useful_name, size_data
from scipy.spatial import ConvexHull

class UpdaterAddFeatureCreateDataSet(object):

    def __init__(self,manager_stop, distance_matrix, distance_depot, size_ball, per_dense, per_sparse,list_k,list_percentile,stop_added):
        """
        Help to quickly update the features
        :param manager_stop: a manager stop
        :param distance_matrix:
        :param distance_depot:
        :param size_ball:
        :param per_dense:
        :param per_sparse:
        """
        self.stop = stop_added

        assert not self.stop.guid in manager_stop, stop_added.guid
        self.manager_stop = manager_stop
        self.distance_matrix = self._update_dist_matrix(distance_matrix)
        self.distance_depot = self._update_depot_dist(distance_depot)

        # Parameter
        self.SIZE_BALL = size_ball
        self.PERCENTAGE_DENSE = per_dense       # of the average density
        self.PERCENTAGE_SPARSE = per_sparse       # of the average density
        self.list_k =list_k
        self.list_percentile = list_percentile


        # Working object
        self.working_dict_feature = {}
        self.previous_dict_feature = None
        self.features_updated = []                # list of features to be updated


    def _update_dist_matrix(self,dist_matrix):
        """
        :param dist_matrix: the previous dist matrix
        :return: a dist matrix
        """
        # check if we have already encountered this stop
        if not self.stop.guid in dist_matrix:
            dist_matrix[self.stop.guid] = {}

        for other_stop in self.manager_stop.values():
            dist = self.stop.get_distance_to_another_stop(other_stop)
            dist_matrix[self.stop.guid][other_stop.guid] = dist
            dist_matrix[other_stop.guid][self.stop.guid] = dist

        dist_matrix[self.stop.guid][self.stop.guid] =0

        return dist_matrix

    def _update_depot_dist(self,dist_depot):
        """
        :param dist_depot: the preivous dist to depot
        :return: a depot dist matrix
        """
        dist_depot[self.stop.guid] = self.stop.get_distance_to_another_stop(self.manager_stop.depot)
        return dist_depot


    def compute_feature_add_stop(self, dict_feature,list_feature):
        """
        Update the dict feature if we add the stop, on the precised list_feature
        :param dict_feature: the previous dict_feature
        :param list_feature: the specific list of features we want to update
        :return: a dict feature
        """
        self.previous_dict_feature = dict_feature
        list_all = [feat for feat in list_feature if "_all" in feat]
        self._update_add_stop(list_stop=list(self.manager_stop.keys()),guid= "_all",list_features=list_all)

        btw,etw = self.stop.TW
        if etw == 250:
            list_early = []
            list_feat_early = [feat for feat in list_feature if "_early" in feat]
            for stop in self.manager_stop.values():
                btw,etw = stop.TW
                if etw == 250:
                    list_early.append(stop.guid)

            self._update_add_stop(list_stop= list_early,guid= "_early",list_features=list_feat_early)

        elif btw == 750:
            list_late = []
            list_feat_late = [feat for feat in list_feature if "_late" in feat]
            for stop in self.manager_stop.values():
                btw,etw = stop.TW
                if btw == 750:
                    list_late.append(stop.guid)

            self._update_add_stop(list_stop=list_late,guid="_late",list_features=list_feat_late)

        else:
            list_no_tw = []
            list_feat_no_tw = [feat for feat in list_feature if "_no_tw" in feat]
            for stop in self.manager_stop.values():
                btw,etw = stop.TW
                if btw != 750 and etw != 250:
                    list_no_tw.append(stop.guid)

            self._update_add_stop(list_stop=list_no_tw,guid="_no_tw",list_features=list_feat_no_tw)

        return self.working_dict_feature


    def _update_add_stop(self,list_stop, guid,list_features):
        """
        Update all the guid features
        :param guid: the type of features concerned
        :return:
        """
        string_features = "--"
        for feat in list_features:
            string_features += feat + "--"
        if useful_name.DEMAND in string_features or useful_name.NB_STOPS in string_features:
            self._update_stop_features(guid)

        if useful_name.SPARSE in string_features or useful_name.DENSE in string_features or useful_name.AREA in string_features or useful_name.DENSITY in string_features:
            self._update_density_features(list_stop,guid)

        if useful_name.DEPOT in string_features:
            self._update_depot_features(list_stop,guid)

        if useful_name.TSP in string_features:
            self._update_heuristic_feature(list_stop,guid)

        if useful_name.ENTROPY in string_features:
            list_size = []
            for feat in list_features:
                if useful_name.ENTROPY in feat:
                    list_size.append(int(feat.split('_')[1]))

            for k in list_size:
                self._update_dispersion_feature(list_stop,guid,k)

        if useful_name.DIAMETER in string_features:
            self._update_diameter(list_stop,guid)

        if useful_name.DISTANCE in string_features:
            need_k = False
            for feat in list_features:
                if useful_name.DISTANCE in feat:
                    if len(feat.split('_')) >=6:
                        need_k = True
                        break
            self._update_distance_stops_features(list_stop,guid,need_k)

        # Check that we have done all features
        not_treated_feature = set(list_features).difference(self.features_updated)
        assert len(not_treated_feature) ==0, print(not_treated_feature,list_features,string_features)


    def _update_stop_features(self,guid):
        """
        Update all features corresponding to stops caracteristics
        """
        key_name = useful_name.NB_STOPS + guid
        self.working_dict_feature[key_name] = 1 + self.previous_dict_feature[key_name]
        self.features_updated.append(key_name)

        key_name = useful_name.DEMAND + guid
        self.working_dict_feature[key_name] = self.stop.demand + self.previous_dict_feature[key_name]
        self.features_updated.append(key_name)


    def _update_density_features(self,list_stop_id,guid):
        """
        Update all density features
        """
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

        # Update affected stops
        list_affected = [stop_id for stop_id in list_stop_id if self.distance_matrix[stop_id][self.stop.guid] <= self.SIZE_BALL]

        for stop_id in list_affected:
            list_dist = [self.distance_matrix[stop_id][stop_id_2] for stop_id_2 in list_stop_id]
            nb_close = np.count_nonzero(np.array(list_dist) <= self.SIZE_BALL) + 1
            if threshold_dense + 1 > nb_close >= threshold_dense:
                nb_dense +=1

            if threshold_sparse +1 >= nb_close > threshold_sparse:
                nb_sparse -=1

        # check itself
        nb_close = len(list_affected) +1  #+1 since you count yourself
        if nb_close >= threshold_dense:
            nb_dense +=1
        if nb_close <= threshold_sparse:
            nb_sparse +=0
        else:
            nb_sparse -=1

        key_name = "not_" + useful_name.SPARSE + guid
        self.working_dict_feature[key_name] = self.previous_dict_feature[key_name] - nb_sparse
        # print(key_name,self.working_dict_feature[key_name],self.previous_dict_feature[key_name])
        self.features_updated.append(key_name)

        key_name = useful_name.DENSE + guid
        self.working_dict_feature[key_name] = nb_dense + self.previous_dict_feature[key_name]
        self.features_updated.append(key_name)


        spatial_coord = [[self.manager_stop[stop_id].x,self.manager_stop[stop_id].y] for stop_id in list_stop_id]
        spatial_coord.append([self.stop.x,self.stop.y])
        if len(spatial_coord) > 2:
            try:
                hull = ConvexHull(spatial_coord)
                area = hull.volume
                density = (len(list_stop_id) + 1)/area
            except scipy.spatial.qhull.QhullError:
                area = 1
                density = 1

        else:
            area = 0
            density = 0
        key_name = 'area' + guid
        self.working_dict_feature[key_name] = area
        self.features_updated.append(key_name)

        key_name = 'density' + guid
        self.working_dict_feature[key_name] = density
        self.features_updated.append(key_name)


    def _update_depot_features(self,list_stop,guid):
        """
        Update all features related to distance between stops and depot
        :param list_stop: the list of stops considered
        :param guid: the guid associated to this feature
        :return: a dict['features_name'] = feature value
        """
        # Mean of pairwise distance
        key_name = useful_name.DEPOT +"_mean_pairwise" + guid
        self.working_dict_feature[key_name] = (len(list_stop) * self.previous_dict_feature[key_name] + self.distance_depot[self.stop.guid])/ (len(list_stop) + 1)
        self.features_updated.append(key_name)

        # Max of pairwise distance
        key_name = useful_name.DEPOT+"_max_pairwise" + guid
        self.working_dict_feature[key_name] = max(self.distance_depot[self.stop.guid], self.previous_dict_feature[key_name])
        self.features_updated.append(key_name)

        # Min of pairwise distance
        key_name = useful_name.DEPOT+"_min_pairwise" + guid
        if len(list_stop) ==0:
            self.working_dict_feature[key_name] = self.distance_depot[self.stop.guid]
        else:
            self.working_dict_feature[key_name] = min(self.distance_depot[self.stop.guid], self.previous_dict_feature[key_name])
        self.features_updated.append(key_name)


    def _update_heuristic_feature(self,list_stop,guid):
        """
        derive all features related to tour of stops
        :param list_stop: the list of stops considered
        :param guid: the guid associated to this feature
        :return: a dict['features_name'] = feature value
        """
        list_to_treat = list_stop.copy()
        list_to_treat.append(self.stop.guid)

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

        key_name = useful_name.TSP + guid
        self.working_dict_feature[key_name] = tot_dist
        self.features_updated.append(key_name)


    def _update_dispersion_feature(self,list_stop_id, guid,k):
        """
        Derive dispersion features, based on entropy
        :param list_stop_id: the list of stop considered
        :param guid:
        :param k: corresponds to the pixel size
        :return: a dict of features
        """
        # Need to build the pixel matrices
        target_i, _ = divmod(self.stop.x,k)
        target_j, _ = divmod(self.stop.y,k)
        pixel_target =0
        for stop_id in list_stop_id:
            stop = self.manager_stop[stop_id]
            i, _ = divmod(stop.x,k)
            j, _ = divmod(stop.y,k)
            if i == target_i and j == target_j:
                pixel_target += stop.demand

        # Compute entropy:
        if pixel_target == 0:
            remove =0
        else:
            remove = pixel_target * np.log(pixel_target)
        pixel_target += self.stop.demand
        add = pixel_target * np.log(pixel_target)
        key_name = useful_name.ENTROPY + "_" + str(k) + guid
        self.working_dict_feature[key_name] = self.previous_dict_feature[key_name] + add - remove
        self.features_updated.append(key_name)


    def _update_diameter(self,list_stop_id,guid):
        """
        Derive diameter
        :param list_stop_id:
        :param guid:
        :return:
        """
        # Update diameter
        if len(list_stop_id) == 0:
            key_name = useful_name.DIAMETER + guid
            self.working_dict_feature[key_name] = 0
            self.features_updated.append(key_name)

        else:
            serv_time = self.stop.service_time
            list_dist_stop = [self.distance_matrix[self.stop.guid][otherId] + serv_time for otherId in list_stop_id]
            list_dist_stop.extend([self.distance_matrix[otherId][self.stop.guid] + self.manager_stop[otherId].service_time for otherId in list_stop_id])
            key_name = useful_name.DIAMETER + guid
            self.working_dict_feature[key_name] = max(max(list_dist_stop),self.previous_dict_feature[key_name])
            self.features_updated.append(key_name)


    def _update_distance_stops_features(self,list_stop_id,guid,use_k):
        """
        Derive distance feature
        :param list_stop_id: the list of stop considered
        :param guid:
        :param use_k: boolean, indicates if we need to compute the percentile.
        """
        n = len(list_stop_id)
        if n ==0:
            self._update_distance_stops_features_zero(guid,use_k)

        else:
            pairwise_stop_dist = [self.distance_matrix[self.stop.guid][otherId] for otherId in list_stop_id]
            pairwise_stop_dist.extend([self.distance_matrix[otherId][self.stop.guid] for otherId in list_stop_id])

            # Mean of pairwise distance
            key_name = useful_name.DISTANCE +"_mean_pairwise" + guid
            mean_list= ((n-1)*n)*self.previous_dict_feature[key_name] + sum(a for a in pairwise_stop_dist)
            mean_list = mean_list/(n*(n+1))
            self.working_dict_feature[key_name] = mean_list
            self.features_updated.append(key_name)

            # Min of pairwise distance
            key_name = useful_name.DISTANCE +"_min_pairwise" + guid
            if self.previous_dict_feature[key_name] == 0 and len(list_stop_id) ==1:
                min_list = min(pairwise_stop_dist)
            else:
                min_list= min(self.previous_dict_feature[key_name],min(pairwise_stop_dist))
            self.working_dict_feature[key_name] = min_list
            self.features_updated.append(key_name)

            # Max of pairwise distance
            key_name = useful_name.DISTANCE +"_max_pairwise" + guid
            max_list = max(self.previous_dict_feature[key_name],max(pairwise_stop_dist))
            self.working_dict_feature[key_name] = max_list
            self.features_updated.append(key_name)


            # mean of k closest stops
            if use_k:
                dict_closest_k = {min(true_k,len(list_stop_id)) - 1: [] for true_k in self.list_k}
                dict_max_k = {min(true_k,len(list_stop_id)) - 1: [] for true_k in self.list_k}
                for stopId in list_stop_id:
                    serv_time = self.manager_stop[stopId].service_time
                    list_dist = [self.distance_matrix[stopId][otherId] + serv_time for otherId in list_stop_id if otherId != stopId]
                    list_dist.append(self.distance_matrix[stopId][self.stop.guid] + serv_time)
                    list_dist = np.array(list_dist)

                    for true_k in self.list_k:
                        k = min(true_k,len(list_stop_id)) - 1
                        idx_closest = np.argpartition(list_dist,k)
                        dict_closest_k[k].append(np.mean(list_dist[idx_closest[:k+1]]))
                        dict_max_k[k].append(list_dist[idx_closest[k]])

                # Specific to our stop
                serv_time = self.stop.service_time
                list_dist = np.array([self.distance_matrix[self.stop.guid][otherId] + serv_time for otherId in list_stop_id])

                for true_k in self.list_k:
                    k = min(true_k,len(list_stop_id)) - 1
                    idx_closest = np.argpartition(list_dist,k)
                    dict_closest_k[k].append(np.mean(list_dist[idx_closest[:k+1]]))
                    dict_max_k[k].append(list_dist[idx_closest[k]])


                for true_k in self.list_k:
                    k = min(true_k,len(list_stop_id)) - 1

                    for percent in self.list_percentile:
                        key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + '_percentile_' + str(percent) + guid
                        self.working_dict_feature[key_name] =np.percentile(dict_closest_k[k],percent)
                        self.features_updated.append(key_name)
                        key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + '_percentile_' + str(percent) + guid
                        self.working_dict_feature[key_name] = np.percentile(dict_max_k[k],percent)
                        self.features_updated.append(key_name)


    def _update_distance_stops_features_zero(self,guid,use_k):
        """
        Deal with the case where one of zero stops
        """
        # List of features to be set to zero
        list_name = list()
        list_name.append(useful_name.DISTANCE +"_mean_pairwise" + guid)
        list_name.append(useful_name.DISTANCE +"_min_pairwise" + guid)
        list_name.append(useful_name.DISTANCE +"_max_pairwise" + guid)

        if use_k:
            for true_k in self.list_k:
                for percent in self.list_percentile:
                    list_name.append(useful_name.DISTANCE +"_closest_mean_" + str(true_k) + '_percentile_' + str(percent) + guid)
                    list_name.append(useful_name.DISTANCE +"_closest_max" + str(true_k) + '_percentile_' + str(percent) + guid)

        for name in list_name:
            self.working_dict_feature[name] = 0
            self.features_updated.append(name)
