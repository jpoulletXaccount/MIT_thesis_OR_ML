
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset,derivingFeaturesCreatedDatasetLinear

import ast
import pandas as pd
import numpy as np

class Cluster(stops_manager_cvrptw.StopsManagerCVRPTW):
    """
    Instancie a cluster class
    """
    def __init__(self,depot,guid,tree,dict_cst,dict_disp):
        super(Cluster,self).__init__(depot)
        self.guid = guid
        self.dict_cst = dict_cst
        self.dict_disp = dict_disp
        self.tree = tree
        self.prediction = None
        self.expected_prediction = None
        self.dict_features = None
        self.leaf = None
        self.featurer = None
        self._centroid = None

        # Tracking how it evolved
        self.tracking_evolution = []

    @property
    def centroid(self):
        if not self._centroid is None:
            return self._centroid
        else:
            x_mean = np.mean([stop.x for stop in self.values()])
            y_mean = np.mean([stop.y for stop in self.values()])
            self._centroid = (x_mean,y_mean)
            return x_mean,y_mean

    @classmethod
    def from_manager_stops(cls,manager_stop,guid,tree,dict_cst,list_useful_features,dict_disp):
        """
        Create a cluster from a manger stop
        :param manager_stop: the manager considered
        :param guid: the guid of the cluster
        :param tree: a tree object
        :param dict_cst: a dict of constraints
        :param list_useful_features: a list of features
        :return: a cluster object
        """
        new_cluster = cls(depot=manager_stop.depot,
                          guid=guid,
                          tree= tree,
                          dict_cst=dict_cst,
                          dict_disp=dict_disp)
        new_cluster.matrix_stops_dist = manager_stop.matrix_stops_dist
        new_cluster.matrix_depot_dist = manager_stop.matrix_depot_dist

        for stop_id in manager_stop.keys():
            new_cluster[stop_id] = manager_stop[stop_id]

        # new_cluster.featurer = derivingFeaturesCreatedDatasetLinear.DerivingFeaturesCreatedDatasetLinear(new_cluster,list_useful_features)
        new_cluster.featurer = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(new_cluster,list_useful_features)
        new_cluster.dict_features = new_cluster.featurer.derive_features()

        # df_feature = [list(new_cluster.dict_features.values())]
        # index_leaf = tree.apply(df_feature)[0]
        index_leaf = new_cluster._get_leaf(new_cluster.dict_features)
        label = tree.get_classification_label(index_leaf)

        # Make sure that we don't sky rocket in terms of number of vehicles.
        if label >= 4.5:
            label = 1000
            expected_label = 1000
        else:
            expected_label = sum(int(nb_vehicle) * proba for nb_vehicle, proba in tree.get_classification_proba(index_leaf).items())

        new_cluster.prediction = label
        new_cluster.expected_prediction = expected_label
        new_cluster.leaf = index_leaf

        return new_cluster

    @classmethod
    def from_row(cls,row,manager_ref,tree,dict_cst,list_useful_features,dict_disp):
        """
        Create a cluster from a row
        :param row: the row from which we read the stop
        :param manager_ref: the manager reference
        :param tree: a tree object
        :param dict_cst: a dict of constraints
        :param list_useful_features: a list of features
        :return: a cluster object
        """
        new_cluster = cls(depot=manager_ref.depot,
                          guid=row['guid'],
                          tree= tree,
                          dict_cst=dict_cst,
                          dict_disp=dict_disp)
        new_cluster.matrix_stops_dist = manager_ref.matrix_stops_dist
        new_cluster.matrix_depot_dist = manager_ref.matrix_depot_dist

        list_stop = ast.literal_eval(row['list_stop'])
        for stop_id in list_stop:
            new_cluster[stop_id] = manager_ref[stop_id]

        # new_cluster.featurer = derivingFeaturesCreatedDatasetLinear.DerivingFeaturesCreatedDatasetLinear(new_cluster,list_useful_features)
        new_cluster.featurer = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(new_cluster,list_useful_features)
        new_cluster.dict_features = {}
        for c in row.keys():
            if not c in ['guid','list_stop','tracking_evolution']:
                new_cluster.dict_features[c] = row[c]

        # df_feature = pd.DataFrame([new_cluster.dict_features])
        # index_leaf = tree.apply(df_feature)[0]
        index_leaf = new_cluster._get_leaf(new_cluster.dict_features)
        label = tree.get_classification_label(index_leaf)

        # Make sure that we don't sky rocket in terms of number of vehicles.
        if label >= 4.5:
            label = 1000
            expected_label = 1000
        else:
            expected_label = sum(int(nb_vehicle) * proba for nb_vehicle, proba in tree.get_classification_proba(index_leaf).items())

        new_cluster.prediction = label
        new_cluster.expected_prediction = expected_label
        new_cluster.leaf = index_leaf
        new_cluster.tracking_evolution = ast.literal_eval(row['tracking_evolution'])

        return new_cluster


    def update_features(self,dict_feat,should_be_respected):
        """
        Update the previous dict of features
        """
        self.dict_features.update(dict_feat)
        index_leaf = self._get_leaf(self.dict_features)
        if index_leaf !=self.leaf and should_be_respected:
            list_cst = self.dict_cst[str(self.leaf)]
            for cst in list_cst:
                if not cst.is_respected(self.dict_features):
                    print('cst is not respected ',cst.to_print)
            print(index_leaf, self.leaf)
            df_feature = [list(self.dict_features.values())]
            assert self.tree.apply(df_feature)[0] == index_leaf
            assert False

    def _get_leaf(self,dict_features):
        """
        return the leaf of the tree
        :param dict_features:
        :return:
        """
        for leaf_id in self.dict_cst:
            list_cst = self.dict_cst[leaf_id]
            has_found = True

            for cst in list_cst:
                if not cst.is_respected(dict_features):
                    has_found = False
                    break

            if has_found:
                return int(leaf_id)

        else:
            assert False

    def reset_features(self,already_updated):
        """
        Rederived the features for the specific cluster
        :already_updated: a list of features which are already updtodate. No need to redo them.
        :return:
        """
        assert len(self.featurer.manager_stop) == len(self)
        list_features_to_compute = [feat for feat in self.featurer.list_useful_features if not feat in already_updated]
        updated_dict = self.featurer.compute_specific_list_feature(list_features_to_compute)
        self.dict_features.update(updated_dict)
        # df_feature = [list(self.dict_features.values())]
        self.leaf = self._get_leaf(self.dict_features)
        label = self.tree.get_classification_label(self.leaf)

        # Make sure that we don't sky rocket in terms of number of vehicles.
        if label >= 4.5:
            label = 1000
            expected_label = 1000
        else:
            expected_label = sum(int(nb_vehicle) * proba for nb_vehicle, proba in self.tree.get_classification_proba(self.leaf).items())

        self.prediction = label
        self.expected_prediction = expected_label

        # reset centoid
        x_mean = np.mean([stop.x for stop in self.values()])
        y_mean = np.mean([stop.y for stop in self.values()])
        self._centroid = x_mean,y_mean


    def set_prediction(self,predict):
        """
        :param predict: the class label
        :return:
        """
        self.prediction = predict

    def set_expected_prediction(self, expected):
        """
        :param expected: the expectation of the class label
        :return:
        """
        self.expected_prediction = expected

    def set_expected_prediction_via_proba(self, dict_proba):
        """
        :param dict_proba: the proba for each class
        :return:
        """
        final_class = 0
        for label,proba in dict_proba.items():
            final_class += label * proba
        self.expected_prediction = final_class


    def add_stop(self,stop):
        """
        Add a stop to the clusters
        :param stop: the stop to be added
        :return:
        """
        assert not stop.guid in self.keys(), stop.guid
        self[stop.guid] = stop
        assert stop.guid in self.featurer.manager_stop, stop.guid


    def remove_stop(self,stop):
        """
        :param stop: the stop to be remove from the clusters
        :return:
        """
        assert stop.guid in self.keys(), stop.guid + str(self.keys())
        del self[stop.guid]
        assert not stop.guid in self.featurer.manager_stop, stop.guid


    def to_dict(self):
        """
        :return: a dict which is useful to re-create the clusters
        """
        row = dict()
        row['guid'] = self.guid
        row['list_stop'] = str(list(self.keys()))
        row['tracking_evolution'] = str(self.tracking_evolution)
        row.update(self.dict_features)
        return row


    def is_robust(self,threshold):
        """
        Check if the cluster is considered as robust given the considered threshold
        :param threshold: the threshold considered
        :return: a boolean
        """
        cst_cluster =self.dict_cst[str(self.leaf)]
        is_robust = True
        for cst in cst_cluster:
            feat_name = cst.get_feature_name(self.dict_features)
            threshold = self.dict_disp[feat_name] * threshold
            if not cst.is_respected(self.dict_features,threshold):
                is_robust = False
                break
        return is_robust

