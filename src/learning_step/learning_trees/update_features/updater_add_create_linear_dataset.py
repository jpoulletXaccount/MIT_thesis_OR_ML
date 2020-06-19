

from src.helpers import useful_name

class UpdaterAddFeatureLinearCreateDataSet(object):

    def __init__(self,manager_stop,distance_depot, stop_added):
        """
        Help to quickly update the features
        :param manager_stop: a manager stop
        """
        self.stop = stop_added

        assert not self.stop.guid in manager_stop, stop_added.guid
        self.manager_stop = manager_stop
        self.distance_depot = self._update_depot_dist(distance_depot)

        # Working object
        self.working_dict_feature = {}
        self.previous_dict_feature = None
        self.features_updated = []                # list of features to be updated


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
        self._update_add_stop(guid= "_all",list_features=list_all)

        btw,etw = self.stop.TW
        if etw == 250:
            list_feat_early = [feat for feat in list_feature if "_early" in feat]
            self._update_add_stop(guid= "_early",list_features=list_feat_early)

        elif btw == 750:
            list_feat_late = [feat for feat in list_feature if "_late" in feat]
            self._update_add_stop(guid="_late",list_features=list_feat_late)

        else:
            list_feat_no_tw = [feat for feat in list_feature if "_no_tw" in feat]
            self._update_add_stop(guid="_no_tw",list_features=list_feat_no_tw)

        return self.working_dict_feature


    def _update_add_stop(self, guid,list_features):
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

        if useful_name.DEPOT in string_features:
            self._update_depot_features(guid)


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


    def _update_depot_features(self,guid):
        """
        Update all features related to distance between stops and depot
        :param guid: the guid associated to this feature
        :return: a dict['features_name'] = feature value
        """
        # Sum of pairwise distance
        key_name = useful_name.DEPOT +"_sum_dist" + guid
        self.working_dict_feature[key_name] = self.previous_dict_feature[key_name] + self.distance_depot[self.stop.guid]
        self.features_updated.append(key_name)
