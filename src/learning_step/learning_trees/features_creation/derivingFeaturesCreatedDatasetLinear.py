
from src.helpers import useful_name
from src.learning_step.learning_trees.update_features import updater_add_create_linear_dataset,updater_remove_create_linear_dataset

class DerivingFeaturesCreatedDatasetLinear(object):
    """
    Class which derives features relevant to the created dataset
    """

    def __init__(self,manager_stop,list_useful_features = None):
        self.manager_stop = manager_stop
        self.distance_depot = manager_stop.matrix_depot_dist
        self.features_updated = []

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
            all_features.extend([useful_name.NB_STOPS + guid,useful_name.DEMAND + guid,useful_name.DEPOT +"_sum_dist" + guid])

        return all_features


    def derive_features(self,for_mip=False):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        dict_features.update(self.derive_depot_features(list(self.manager_stop.keys()),guid="_all",for_mip=for_mip))
        dict_features.update(self.derive_stop_features(list(self.manager_stop.keys()),guid="_all",for_mip=for_mip))
        dict_features.update(self.derive_TW_features(for_mip=for_mip))

        return dict_features


    def derive_stop_features(self,list_stop_id,guid,for_mip=False):
        """
        Derive all features corresponding to stops caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.NB_STOPS + guid
        self.features_updated.append(key_name)
        if for_mip:
            dict_features[key_name] = list_stop_id
        else:
            dict_features[key_name] = len(list_stop_id)

        key_name = useful_name.DEMAND + guid
        self.features_updated.append(key_name)
        if for_mip:
            dict_features[key_name] = list_stop_id
        else:
            dict_features[key_name] = sum(self.manager_stop[stopId].demand for stopId in list_stop_id)

        return dict_features


    def derive_depot_features(self,list_stop,guid,for_mip=False):
        """
        derive all features related to distance between stops and depot
        :param list_stop: the list of stops considered
        :param guid: the guid associated to this feature
        :param for_mip: indicates if the purpose is to use a MIP or not.
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        if len(list_stop) ==0:
            key_name = useful_name.DEPOT +"_sum_dist" + guid
            self.features_updated.append(key_name)
            if for_mip:
                dict_features[key_name] = []
            else:
                dict_features[key_name] = 0
            return dict_features

        # sum of pairwise distance
        sum_pairwise= sum([self.distance_depot[stopId] for stopId in list_stop])
        key_name = useful_name.DEPOT +"_sum_dist" + guid
        self.features_updated.append(key_name)
        if for_mip:
            dict_features[key_name] = list_stop
        else:
            dict_features[key_name] = sum_pairwise

        return dict_features



    def derive_TW_features(self,for_mip):
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


        dict_features.update(self.derive_depot_features(list_bucket_early,guid="_early",for_mip=for_mip))
        dict_features.update(self.derive_stop_features(list_bucket_early,guid="_early",for_mip=for_mip))

        dict_features.update(self.derive_depot_features(list_bucket_late,guid="_late",for_mip=for_mip))
        dict_features.update(self.derive_stop_features(list_bucket_late,guid="_late",for_mip=for_mip))

        dict_features.update(self.derive_depot_features(list_bucket_no_tw,guid="_no_tw",for_mip=for_mip))
        dict_features.update(self.derive_stop_features(list_bucket_no_tw,guid="_no_tw",for_mip=for_mip))

        return dict_features


    def update_feature_add_stop(self,dict_feature, stop, list_feat):
        """
        Update the dict feature if we add the stop
        :param dict_feature: the previous dict_feature
        :param stop: the stop to be added
        :param list_feat: the list of features we want to update
        :return: a dict feature
        """
        update = updater_add_create_linear_dataset.UpdaterAddFeatureLinearCreateDataSet(manager_stop=self.manager_stop,
                                                                    distance_depot=self.distance_depot,
                                                                    stop_added=stop)
        new_dict_featuer = update.compute_feature_add_stop(dict_feature.copy(),list_feat)
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
        update = updater_remove_create_linear_dataset.UpdaterRemoveFeatureLinearCreateDataSet(manager_stop=self.manager_stop,
                                                                                 distance_depot=self.distance_depot,
                                                                                 stop_removed=stop)
        new_dict_featuer = update.compute_feature_remove_stop(dict_feature.copy(),list_feat)
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

        elif useful_name.DEPOT in feature_name:
            return self.derive_depot_features(list_considered,"_" +guid)

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

        if useful_name.DEPOT in string_features:
            dict_features.update(self.derive_depot_features(list_stop,guid))


        # Check that we have done all features
        not_treated_feature = set(list_features).difference(self.features_updated)
        assert len(not_treated_feature) ==0, print(not_treated_feature,list_features,string_features)

        return dict_features



