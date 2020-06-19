from src.helpers import useful_name,size_data

class DerivingFeaturesPixel(object):
    """
    Class used to derive the useful features
    """

    def __init__(self,manager_pixel):
        self.manager_pixel = manager_pixel


    def derive_features(self):
        """
        Main function of this class, derives all the wanted features and return them
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        dict_features.update(self.derive_TW_features())
        dict_features.update(self.derive_Demand_features())
        dict_features.update(self.derive_Stops_features())

        return dict_features


    def derive_Stops_features(self):
        """
        Derive all features corresponding to stops caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        for pixel in self.manager_pixel.values():
            key_name = useful_name.NB_STOPS + "_" + pixel.guid
            dict_features[key_name] = pixel.number_stop

        return dict_features


    def derive_TW_features(self):
        """
        Derive all features corresponding to TW caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}

        nb_class = 6
        TW_all = size_data.TW
        inter = TW_all[1] - TW_all[0]
        inter_adv = inter/nb_class

        total_number = 0
        for i in range(0,nb_class):
            begin_TW = TW_all[0] + i *inter_adv
            end_TW = TW_all[0] + (i+1) * inter_adv
            for pixel in self.manager_pixel.values():

                feature_name = useful_name.NB_STOPS_BEFORE + str(int(end_TW)) + "_" + str(pixel.guid)
                num = pixel.number_stop_due_in_TW((begin_TW,end_TW))
                dict_features[feature_name] = num
                total_number += num

        assert total_number == self.manager_pixel.number_stops, str(total_number) + "_" + str(self.manager_pixel.number_stops)

        return dict_features


    def derive_Demand_features(self):
        """
        Derive all features corresponding to demand caracteristics
        :return: a dict['features_name'] = feature value
        """
        dict_features = {}
        for pixel in self.manager_pixel.values():
            key_name = useful_name.DEMAND + "_" + pixel.guid
            dict_features[key_name] = pixel.demand

        return dict_features
