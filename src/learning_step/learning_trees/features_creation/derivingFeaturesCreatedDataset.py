import numpy as np
import scipy
from scipy.spatial import ConvexHull

from src.learning_step.learning_trees.update_features import updater_add_create_dataset, updater_remove_create_dataset
from src.helpers import useful_name, size_data

class DerivingFeaturesCreatedDataset(object):
    """
    Class which derives features relevant to the created dataset
    """

    def __init__(self,manager_stop,list_useful_features = None):
        self.manager_stop = manager_stop
        self.distance_matrix = manager_stop.matrix_stops_dist
        self.distance_depot = manager_stop.matrix_depot_dist
        self.features_updated = []

        # Parameter
        self.SIZE_BALL = 10
        self.PERCENTAGE_DENSE = 0.90        # of the average density
        self.PERCENTAGE_SPARSE = 0.50       # of the average density

        self.list_k = [1,2,10]
        self.list_pixel_size = [5,10,15]
        self.list_percentile = [25,50,75,90,100]


        if list_useful_features is None:
            self.list_useful_features = self.get_all_potential_features()
        else:
            self.list_useful_features = list_useful_features


    def get_all_potential_features(self):
        """
        :return: the list of all existing features
        """
        all_features = []
        list_guid = ['_all','_early','_no_tw','_late']
        for guid in list_guid:
            all_features.extend([useful_name.NB_STOPS + guid,useful_name.DEMAND + guid,useful_name.DISTANCE +"_mean_pairwise" + guid,useful_name.DISTANCE +"_min_pairwise" + guid,useful_name.DISTANCE +"_max_pairwise" + guid,
                                 useful_name.DIAMETER + guid,useful_name.TSP + guid])
            all_features.extend([useful_name.DISTANCE +"_closest_mean_" + str(k) + '_percentile_' + str(percent) + guid for k in self.list_k for percent in self.list_percentile])
            all_features.extend([useful_name.ENTROPY + "_" + str(size) + guid for size in self.list_pixel_size])
            all_features.extend(["not_" + useful_name.SPARSE + guid,useful_name.DENSE + guid,useful_name.AREA + guid, useful_name.DENSITY + guid])
            all_features.extend([useful_name.DEPOT +"_min_pairwise" + guid, useful_name.DEPOT +"_mean_pairwise" + guid,useful_name.DEPOT +"_max_pairwise" + guid])

        return all_features

    def derive_features(self):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = feature value
        """
        dict_features = self._derive_all_features(list_stop_id=list(self.manager_stop.keys()),guid="_all")
        dict_features.update(self.derive_TW_features())

        return dict_features

    def _derive_all_features(self,list_stop_id,guid):
        """
        Derive all features and return a dict
        :param list_stop_id: the considered list of stop
        :param guid: the guid
        :return: a dict of features
        """
        dict_features = {}
        dict_features.update(self.derive_density_feature(list_stop_id,guid=guid))
        dict_features.update(self.derive_spatial_features(list_stop_id,guid=guid))
        dict_features.update(self.derive_depot_features(list_stop_id,guid=guid))
        dict_features.update(self.derive_heuristic_feature(list_stop_id,guid= guid))
        dict_features.update(self.derive_stop_features(list_stop_id,guid=guid))
        dict_features.update(self.derive_distance_stops_features(list_stop_id,guid=guid))
        dict_features.update(self.derive_diameter_stops_features(list_stop_id,guid=guid))
        dict_features.update(self.derive_dispersion_feature(list_stop_id,guid=guid))
        dict_features.update(self.derive_k_closest_stops_features(list_stop_id,guid=guid))

        return dict_features

    def derive_stop_features(self,list_stop_id,guid):
        """
        Derive all features corresponding to stops caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.NB_STOPS + guid
        dict_features[key_name] = len(list_stop_id)
        self.features_updated.append(key_name)

        key_name = useful_name.DEMAND + guid
        dict_features[key_name] = sum(self.manager_stop[stopId].demand for stopId in list_stop_id)
        self.features_updated.append(key_name)

        return dict_features


    def derive_distance_stops_features(self,list_stop_id,guid):
        """
        derive all features related to distance between stops
        :return: a dict['features_name'] = feature value
        """
        solved_features = [useful_name.DISTANCE +"_mean_pairwise" + guid, useful_name.DISTANCE +"_min_pairwise" + guid,useful_name.DISTANCE +"_max_pairwise" + guid]
        test_features = [feat for feat in solved_features if feat in self.list_useful_features]
        if len(list_stop_id) <=1 or len(test_features) <=0.5:
            return self._derive_distance_stops_features(guid)

        dict_features = {}
        manager_considered = {stopId : self.manager_stop[stopId] for stopId in list_stop_id}

        pairwise_dist = [self.distance_matrix[stopId][otherId] for stopId in manager_considered.keys() for otherId in manager_considered.keys() if otherId != stopId]

        # Mean of pairwise distance
        mean_list= np.mean(pairwise_dist)
        key_name = useful_name.DISTANCE +"_mean_pairwise" + guid
        dict_features[key_name] = mean_list
        self.features_updated.append(key_name)

        # Min of pairwise distance
        min_list= min(pairwise_dist)
        key_name = useful_name.DISTANCE +"_min_pairwise" + guid
        dict_features[key_name] = min_list
        self.features_updated.append(key_name)

        # Max of pairwise distance
        max_list = max(pairwise_dist)
        key_name = useful_name.DISTANCE +"_max_pairwise" + guid
        dict_features[key_name] = max_list
        self.features_updated.append(key_name)

        return dict_features

    def derive_diameter_stops_features(self,list_stop_id,guid):
        """
        Derive feautres related to the diameter
        :return: dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.DIAMETER + guid

        if len(list_stop_id) <=1 or not key_name in self.list_useful_features:
            dict_features[key_name] = 0

        else:
            manager_considered = {stopId : self.manager_stop[stopId] for stopId in list_stop_id}
            pairwise_dist = [self.distance_matrix[stopId][otherId] + manager_considered[stopId].service_time for stopId in manager_considered.keys() for otherId in manager_considered.keys() if otherId != stopId]
            dict_features[key_name] = max(pairwise_dist)

        self.features_updated.append(key_name)
        return dict_features

    def derive_k_closest_stops_features(self,list_stop_id,guid):
        """
        Derive all features related to the k closest
        :return: a dict['features_name'] = feature value
        """
        solved_features = [useful_name.DISTANCE +"_closest_mean_" + str(k) + '_percentile_' + str(percent) + guid for k in self.list_k for percent in self.list_percentile]
        solved_features.extend([useful_name.DISTANCE +"_closest_max" + str(k) + '_percentile_' + str(percent) + guid for k in self.list_k for percent in self.list_percentile])
        test_features = [feat for feat in solved_features if feat in self.list_useful_features]
        if len(list_stop_id) <=1 or len(test_features) <= 0.5:
            return self._derive_k_closest_stops_features(guid)

        # mean of k closest stops
        dict_features = {}
        manager_considered = {stopId : self.manager_stop[stopId] for stopId in list_stop_id}

        dict_closest_k = {min(true_k,len(list_stop_id) - 1) - 1: [] for true_k in self.list_k}
        dict_max_k = {min(true_k,len(list_stop_id) - 1) - 1: [] for true_k in self.list_k}
        for stopId in manager_considered:
            serv_time = manager_considered[stopId].service_time
            list_dist = np.array([self.distance_matrix[stopId][otherId] + serv_time for otherId in manager_considered.keys() if otherId != stopId])

            for true_k in self.list_k:
                k = min(true_k,len(list_stop_id) - 1) - 1
                idx_closest = np.argpartition(list_dist,k)
                dict_closest_k[k].append(np.mean(list_dist[idx_closest[:k+1]]))
                dict_max_k[k].append(list_dist[idx_closest[k]])


        for true_k in self.list_k:
            k = min(true_k,len(list_stop_id) - 1) - 1

            for percent in self.list_percentile:
                key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] =np.percentile(dict_closest_k[k],percent)
                self.features_updated.append(key_name)
                key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] = np.percentile(dict_max_k[k],percent)
                self.features_updated.append(key_name)


        return dict_features


    def _derive_distance_stops_features(self,guid):
        """
        Second the derive distance stops features in the case of null list
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        key_name = useful_name.DISTANCE +"_mean_pairwise" + guid
        dict_features[key_name] = 0
        self.features_updated.append(key_name)

        key_name = useful_name.DISTANCE +"_min_pairwise" + guid
        dict_features[key_name] = 0
        self.features_updated.append(key_name)

        key_name = useful_name.DISTANCE +"_max_pairwise" + guid
        dict_features[key_name] = 0
        self.features_updated.append(key_name)

        return dict_features


    def _derive_k_closest_stops_features(self,guid):
        """
        Second the derive distance stops features in the case of null list
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

         # mean of k closest stops
        for true_k in self.list_k:
            for percent in self.list_percentile:
                key_name = useful_name.DISTANCE +"_closest_mean_" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] = 0
                self.features_updated.append(key_name)
                key_name = useful_name.DISTANCE +"_closest_max" + str(true_k) + '_percentile_' + str(percent) + guid
                dict_features[key_name] = 0
                self.features_updated.append(key_name)

        return dict_features


    def derive_dispersion_feature(self,list_stop_id, guid):
        """
        Derive dispersion features, based on entropy
        :param list_stop_id: the list of stop considered
        :param guid:
        :return: a dict of features
        """
        dict_features = {}
        # Need to build the pixel matrices
        for size in self.list_pixel_size:
            key_name = useful_name.ENTROPY + "_" + str(size) + guid
            if not key_name in self.list_useful_features:
                entropy = 0
            else:
                nb = int(300/size)
                pixel_mat = np.zeros(shape=(nb,nb))
                for stop_id in list_stop_id:
                    stop = self.manager_stop[stop_id]
                    i, _ = divmod(stop.x,size)
                    j, _ = divmod(stop.y,size)
                    pixel_mat[i,j] += stop.demand

                # Compute entropy:
                entropy = sum(pixel_mat[i,j] * np.log(pixel_mat[i,j]) for i in range(0,nb) for j in range(0,nb) if pixel_mat[i,j] !=0)

            dict_features[key_name] = entropy
            self.features_updated.append(key_name)

        return dict_features


    def derive_density_feature(self,list_stop_id,guid):
        """
        Derive the density feature
        :param list_stop_id: the list of stop to be considered
        :param guid: the guid of this feature
        :return: a dict['features_name'] = feature value
        """
        solved_features= ["not_" + useful_name.SPARSE + guid,useful_name.DENSE + guid]
        test_features = [feat for feat in solved_features if feat in self.list_useful_features]
        dict_features = {}
        if len(list_stop_id) ==0 or len(test_features) <= 0.5:
            key_name = "not_" +useful_name.SPARSE + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)

            key_name = useful_name.DENSE + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)

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

        for stop_id_1 in list_stop_id:
            list_dist = [self.distance_matrix[stop_id_1][stop_id_2] for stop_id_2 in list_stop_id]
            nb_close = np.count_nonzero(np.array(list_dist) <= self.SIZE_BALL)
            if nb_close >= threshold_dense:
                nb_dense +=1
            if nb_close <= threshold_sparse:
                nb_sparse +=1

        key_name = "not_" + useful_name.SPARSE + guid
        dict_features[key_name] = len(list_stop_id) - nb_sparse
        self.features_updated.append(key_name)

        key_name = useful_name.DENSE + guid
        dict_features[key_name] = nb_dense
        self.features_updated.append(key_name)

        return dict_features


    def derive_spatial_features(self,list_stop_id,guid):
        """
        Derive the spatial feature
        :param list_stop_id: the list of stop to be considered
        :param guid: the guid of this feature
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        solved_features= [useful_name.AREA + guid, useful_name.DENSITY + guid]
        test_features = [feat for feat in solved_features if feat in self.list_useful_features]
        if len(list_stop_id) ==0 or len(test_features) <=0.5:
            key_name = useful_name.AREA + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)

            key_name = useful_name.DENSITY + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)

            return dict_features

        spatial_coord = [[self.manager_stop[stop_id].x,self.manager_stop[stop_id].y] for stop_id in list_stop_id]
        if len(spatial_coord) > 2:
            try:
                hull = ConvexHull(spatial_coord)
                area = hull.volume
                density = len(list_stop_id)/area
            except scipy.spatial.qhull.QhullError:
                area = 1
                density = 1

        else:
            area = 0
            density = 0
        key_name = useful_name.AREA + guid
        dict_features[key_name] = area
        self.features_updated.append(key_name)

        key_name = useful_name.DENSITY + guid
        dict_features[key_name] = density
        self.features_updated.append(key_name)

        return dict_features


    def derive_depot_features(self,list_stop,guid):
        """
        derive all features related to distance between stops and depot
        :param list_stop: the list of stops considered
        :param guid: the guid associated to this feature
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        solved_features= [useful_name.DEPOT +"_min_pairwise" + guid, useful_name.DEPOT +"_mean_pairwise" + guid,useful_name.DEPOT +"_max_pairwise" + guid]
        test_features = [feat for feat in solved_features if feat in self.list_useful_features]
        if len(list_stop) ==0 or len(test_features) <= 0.5:
            key_name = useful_name.DEPOT +"_min_pairwise" + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)
            key_name = useful_name.DEPOT +"_mean_pairwise" + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)
            key_name = useful_name.DEPOT+"_max_pairwise" + guid
            dict_features[key_name] = 0
            self.features_updated.append(key_name)
            return dict_features

        dist_depot = [self.distance_depot[stopId] for stopId in list_stop]

        # Min of pairwise distance
        min_pairwise= min(dist_depot)
        key_name = useful_name.DEPOT +"_min_pairwise" + guid
        dict_features[key_name] = min_pairwise
        self.features_updated.append(key_name)

        # Mean of pairwise distance
        mean_pairwise= np.mean(dist_depot)
        key_name = useful_name.DEPOT +"_mean_pairwise" + guid
        dict_features[key_name] = mean_pairwise
        self.features_updated.append(key_name)

        # Max of pairwise distance
        max_depot= max(dist_depot)
        key_name = useful_name.DEPOT+"_max_pairwise" + guid
        dict_features[key_name] = max_depot
        self.features_updated.append(key_name)

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
        if len(list_stop) == 0 or not key_name in self.list_useful_features:
            dict_features[key_name] = 0
            self.features_updated.append(key_name)
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
        self.features_updated.append(key_name)
        return dict_features


    def derive_TW_features(self):
        """
        derive features by TW bucket
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        list_bucket_early = []
        list_bucket_late = []
        list_bucket_no_tw = []

        for stop in self.manager_stop.values():
            btw,etw = stop.TW
            if etw == 250:
                list_bucket_early.append(stop.guid)
            elif btw == 750:
                list_bucket_late.append(stop.guid)
            else:
                list_bucket_no_tw.append(stop.guid)

        dict_features.update(self._derive_all_features(list_bucket_early,guid="_early"))
        dict_features.update(self._derive_all_features(list_bucket_late,guid="_late"))
        dict_features.update(self._derive_all_features(list_bucket_no_tw,guid="_no_tw"))

        return dict_features


    def update_feature_add_stop(self,dict_feature, stop, list_feat):
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


    def update_feature_remove_stop(self,dict_feature, stop, list_feat):
        """
        Update the dict feature if we remove the stop
        :param dict_feature: the previous dict_feature
        :param stop: the stop to be removed
        :param list_feat: the list of features we want to update
        :return: a dict feature
        """
        update = updater_remove_create_dataset.UpdaterRemoveFeatureCreateDataSet(manager_stop=self.manager_stop,
                                                                                 distance_matrix=self.distance_matrix,
                                                                                 distance_depot=self.distance_depot,
                                                                                 size_ball=self.SIZE_BALL,
                                                                                 per_dense=self.PERCENTAGE_DENSE,
                                                                                 per_sparse=self.PERCENTAGE_SPARSE,
                                                                                 list_k=self.list_k,
                                                                                 list_percentile=self.list_percentile,
                                                                                 stop_removed=stop)
        new_dict_featuer = update.compute_feature_remove_stop(dict_feature.copy(),list_feat)
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

        elif "no_tw" in guid:
            list_bucket_no_tw = []
            for stop in self.manager_stop.values():
                btw,etw = stop.TW
                if btw != 750 and etw != 250:
                    list_bucket_no_tw.append(stop.guid)
                list_considered = list_bucket_no_tw

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


    def compute_specific_list_feature(self,list_features):
        """
        :param list_features: the list of the features to be computed
        :return: the value
        """
        list_bucket_no_tw = []
        list_bucket_early = []
        list_bucket_late = []

        for stop in self.manager_stop.values():
            btw,etw = stop.TW
            if etw == 250:
                list_bucket_early.append(stop.guid)
            elif btw == 750:
                list_bucket_late.append(stop.guid)
            else:
                list_bucket_no_tw.append(stop.guid)

        list_feat_early = [feat for feat in list_features if '_early' in feat]
        list_feat_all = [feat for feat in list_features if '_all' in feat]
        list_feat_late = [feat for feat in list_features if '_late' in feat]
        list_feat_no_tw = [feat for feat in list_features if '_no_tw' in feat]
        assert len(list_feat_no_tw + list_feat_early + list_feat_all + list_feat_late) == len(list_features)

        new_dict_feature = self._update_list_features(list(self.manager_stop.keys()),'_all',list_feat_all)
        new_dict_feature.update(self._update_list_features(list_bucket_early,'_early',list_feat_early))
        new_dict_feature.update(self._update_list_features(list_bucket_late,'_late',list_feat_late))
        new_dict_feature.update(self._update_list_features(list_bucket_no_tw,'_no_tw',list_feat_no_tw))

        return new_dict_feature


    def _update_list_features(self,list_stop,guid,list_features):
        """
        Update a specific list of features
        :param list_stop: the corresponding list stop
        :param guid: indicates which category of tw treated
        :param list_features: the lsit of features we should focus on
        :return: the new dict features
        """
        self.features_updated = list()

        string_features = "--"
        for feat in list_features:
            string_features += feat + "--"

        dict_features = {}
        if useful_name.DEMAND in string_features or useful_name.NB_STOPS in string_features:
            dict_features.update(self.derive_stop_features(list_stop,guid))

        if useful_name.SPARSE in string_features or useful_name.DENSE in string_features:
            dict_features.update(self.derive_density_feature(list_stop,guid))

        if useful_name.AREA in string_features or useful_name.DENSITY in string_features:
            dict_features.update(self.derive_spatial_features(list_stop,guid))

        if useful_name.DIAMETER in string_features:
            dict_features.update(self.derive_diameter_stops_features(list_stop,guid))

        if 'pairwise' in string_features and useful_name.DISTANCE in string_features:
            dict_features.update(self.derive_distance_stops_features(list_stop,guid))

        if 'closest' in string_features:
            dict_features.update(self.derive_k_closest_stops_features(list_stop,guid))

        if useful_name.ENTROPY in string_features:
            dict_features.update(self.derive_dispersion_feature(list_stop,guid))

        if useful_name.DEPOT in string_features:
            dict_features.update(self.derive_depot_features(list_stop,guid))

        if useful_name.TSP in string_features:
            dict_features.update(self.derive_heuristic_feature(list_stop,guid))


        # Check that we have done all features
        not_treated_feature = set(list_features).difference(self.features_updated)
        assert len(not_treated_feature) ==0, print(not_treated_feature,list_features,string_features)

        return dict_features




