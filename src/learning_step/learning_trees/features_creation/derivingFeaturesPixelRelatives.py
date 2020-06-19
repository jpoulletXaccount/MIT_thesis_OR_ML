from src.helpers import useful_name,size_data
from collections import Counter

class DerivingFeaturesPixelRelatives(object):
    """
    Class used to derive the useful features
    """

    def __init__(self,manager_pixel,matrix_pix):
        self.manager_pixel = manager_pixel
        self._matrix_dist = matrix_pix

        # Parameters
        self.DIST_THRESHOLD_CLOSE = 25      # corresponds to 15 min
        self.DIST_THRESHOLD_FAR = 100       # corresponds to one hour
        self.NUMBER_BUCKET_TW = 7           # number of time windows division we do


    def _build_pixel_distance(self):
        """
        :return: a dict[pixel_id][pixel_id] = dist
        """

        list_pixel_id = list(self.manager_pixel.keys())
        dict_matrix = {pixelId : {} for pixelId in list_pixel_id}

        for i in range(0,len(list_pixel_id)):
            pixel_id_1 = list_pixel_id[i]
            pixel_1 = self.manager_pixel[pixel_id_1]

            for j in range(i, len(list_pixel_id)):
                pixel_id_2 = list_pixel_id[j]
                pixel_2 = self.manager_pixel[pixel_id_2]

                dist = pixel_1.get_distance_to_point(pixel_2)
                dict_matrix[pixel_id_1][pixel_id_2] = dist
                dict_matrix[pixel_id_2][pixel_id_1] = dist


        return dict_matrix


    def derive_features(self):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        dict_features.update(self._derive_Demand_features())
        dict_features.update(self._derive_Stops_features())
        dict_features.update(self._derive_depot_features())
        dict_features.update(self._derive_TW_features())

        return dict_features


    def derive_list_stop_id_features(self):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = a counter list_stop    that we should sum
        """
        dict_features = {}
        dict_features.update(self._derive_Demand_features(list_stop = True))
        dict_features.update(self._derive_Stops_features(list_stop = True))
        dict_features.update(self._derive_depot_features(list_stop = True))
        dict_features.update(self._derive_TW_features(list_stop = True))

        return dict_features


    def _derive_depot_features(self,list_stop = False):
        """
        Derive all features corresponding to the depot location. Mainly regarding the early TW
        :param list_stop: if True then isntead of having the value, has the list of stops
        :return: a dict['features_name'] = feature value, or list of stops
        """
        TW_all = size_data.TW
        inter = TW_all[1] - TW_all[0]
        inter_adv = inter/self.NUMBER_BUCKET_TW
        early_TW = inter_adv

        tot_num = 0
        tot_list_stop = []
        for pixel in self.manager_pixel.values():
            # check if far
            if pixel.get_distance_to_point(self.manager_pixel.depot) >= self.DIST_THRESHOLD_FAR:
                new_list_stop = pixel.list_stop_due_in_TW((0,early_TW))
                tot_list_stop.extend(new_list_stop)
                tot_num += len(new_list_stop)

        dict_features = {}
        key_name = useful_name.DEPOT + "_far_early"
        if list_stop:
            dict_features[key_name] = Counter(tot_list_stop)
        else:
            dict_features[key_name] = tot_num

        return dict_features


    def _derive_Stops_features(self,list_stop = False):
        """
        Derive all features corresponding to stops caracteristics
        :param list_stop: if True then isntead of having the value, has the list of stops
        :return: a dict['features_name'] = feature value, or list of stops
        """
        dict_features = {}
        key_name = useful_name.NB_STOPS
        if list_stop:
            dict_features[key_name] = Counter(self.manager_pixel.list_stop_id)
        else:
            dict_features[key_name] = self.manager_pixel.number_stops

        return dict_features


    def _derive_TW_features(self,list_stop = False):
        """
        Derive all features corresponding to TW caracteristics
        :param list_stop: if True then isntead of having the value, has the list of stops
        :return: a dict['features_name'] = feature value,or list of stops
        """
        dict_features = {}

        TW_all = size_data.TW
        inter = TW_all[1] - TW_all[0]
        inter_adv = inter/self.NUMBER_BUCKET_TW

        dict_list_far_away = {}
        dict_list_close = {}
        for i in range(0,self.NUMBER_BUCKET_TW):
            end_TW = TW_all[0] + (i+1) * inter_adv
            dict_list_close[end_TW] = []
            dict_list_far_away[end_TW] = []

        total_list_close = []
        total_list_far = []

        list_pixel_id = list(self.manager_pixel.keys())
        for i in range(0,len(list_pixel_id)):
            pixel1_id = list_pixel_id[i]

            for j in range(i, len(list_pixel_id)):
                pixel2_id = list_pixel_id[j]
                pixel2 = self.manager_pixel[pixel2_id]
                # we count the same pixel as it takes into account the stops inside the pixel which are closed from each others

                if self._matrix_dist[pixel1_id][pixel2.guid]<= self.DIST_THRESHOLD_CLOSE:
                    total_list_close += pixel2.list_stop_guid
                    for k in range(0,self.NUMBER_BUCKET_TW):
                        begin_TW = TW_all[0] + k *inter_adv
                        end_TW = TW_all[0] + (k+1) * inter_adv
                        dict_list_close[end_TW] += pixel2.list_stop_due_in_TW((begin_TW,end_TW))

                elif self._matrix_dist[pixel1_id][pixel2.guid] >= self.DIST_THRESHOLD_FAR:
                    total_list_far += pixel2.list_stop_guid
                    for k in range(0,self.NUMBER_BUCKET_TW):
                        begin_TW = TW_all[0] + k *inter_adv
                        end_TW = TW_all[0] + (k+1) * inter_adv
                        dict_list_far_away[end_TW] += pixel2.list_stop_due_in_TW((begin_TW,end_TW))


        key_name = useful_name.NB_STOPS + "_close"
        if list_stop:
            dict_features[key_name] = Counter(total_list_close)
        else:
            dict_features[key_name] = len(total_list_close)

        key_name = useful_name.NB_STOPS + "_far"
        if list_stop:
            dict_features[key_name] = Counter(total_list_far)
        else:
            dict_features[key_name] = len(total_list_far)

        for endTW in dict_list_far_away:
            feature_name = useful_name.NB_STOPS_BEFORE + str(int(endTW)) + "_far"
            if list_stop:
                dict_features[feature_name] = Counter(dict_list_far_away[endTW])
            else:
                dict_features[feature_name] = len(dict_list_far_away[endTW])

            feature_name = useful_name.NB_STOPS_BEFORE + str(int(endTW)) + "_close"
            if list_stop:
                dict_features[feature_name] = Counter(dict_list_close[endTW])
            else:
                dict_features[feature_name] = len(dict_list_close[endTW])

        return dict_features


    def _derive_Demand_features(self,list_stop = False):
        """
        Derive all features corresponding to demand caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        key_name = useful_name.DEMAND
        if list_stop:
            dict_features[key_name] = Counter(self.manager_pixel.list_stop_id)
        else:
            dict_features[key_name] = self.manager_pixel.total_demand

        return dict_features
