from tqdm.auto import tqdm

from src.optimization_step.scp_approach.objects import cluster
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDataset,derivingFeaturesCreatedDatasetLinear

class ManagerCluster(dict):
    """
    Manager of clusters, group them by their guid
    """
    def __init__(self,manager_stops,tree,dict_cst,dict_disp):
        super(ManagerCluster,self).__init__()
        self.dict_stop_clusters = {stop_id : [] for stop_id in manager_stops.keys()}    # a dict[stop_id] = list cluster id covering it.
        self.tree = tree
        self.dict_cst = dict_cst
        self.list_useful_features = self._derive_list_useful_features(manager_stops,dict_cst)
        self.dict_dispersion = self._derive_dict_dispersion(dict_disp)

        self.last_id_number = 0


    def _derive_dict_dispersion(self,dict_dispersion):
        """
        :return: a clean dict_dispersion, without the useless features
        """
        new_dict = dict()
        for f in self.list_useful_features:
            new_dict[f] = dict_dispersion[f]

        return new_dict

    @staticmethod
    def _derive_list_useful_features(manager_ref,dict_cst):
        """
        From the known cst, derive all the useful
        :param manager_ref: a manger stop
        :param dict_cst: a dict of cst for each leaf
        :return: a list of features
        """
        dump_manager = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist([],manager_ref)
        featurer = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(dump_manager)
        # featurer = derivingFeaturesCreatedDatasetLinear.DerivingFeaturesCreatedDatasetLinear(dump_manager)
        dict_feature = featurer.derive_features()

        list_cst = [cst for l in dict_cst.values() for cst in l]
        # print('Nb constraints ', len(list_cst))
        list_features = [cst.get_feature_name(dict_feature) for cst in list_cst]
        list_features=list(set(list_features))
        # print('nb features ',len(list_features), list_features, 'and all features ', len(dict_feature))

        return list_features


    def build_from_df(self,df,manager_ref):
        """
        Fill in the manager with the cluster contained in the df
        :return:
        """
        for i,row in tqdm(df.iterrows()):
            cluster_build = cluster.Cluster.from_row(row=row,
                                                     manager_ref=manager_ref,
                                                     tree=self.tree,
                                                    dict_cst= self.dict_cst,
                                                     list_useful_features=self.list_useful_features,
                                                     dict_disp=self.dict_dispersion)
            assert not cluster_build.guid in self,cluster_build.guid
            self[cluster_build.guid] = cluster_build
            assert not cluster_build.dict_features is None

            # Update the manager
            for stop_id in cluster_build.keys():
                self.dict_stop_clusters[stop_id].append(cluster_build.guid)

            self.last_id_number +=1


    def create_cluster(self, list_stop_id, manager_ref,tracking):
        """
        From a manager ref, creates the cluster corresponding to the list of stops
        :param list_stop_id: the stops within the clusters
        :param manager_ref: a manager stop of reference
        :param tracking: track how the cluster is created
        :return: the newly guid
        """
        new_guid = self._create_guid()
        new_manager = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(list_stop_id,manager_ref)

        new_cluster = cluster.Cluster.from_manager_stops(manager_stop=new_manager,
                                                         guid=new_guid,
                                                        tree=self.tree,
                                                         dict_cst= self.dict_cst,
                                                         list_useful_features=self.list_useful_features,
                                                         dict_disp=self.dict_dispersion)
        new_cluster.tracking_evolution.append(tracking)
        self[new_guid] = new_cluster
        assert not new_cluster.dict_features is None

        # Update the manager
        for stop_id in list_stop_id:
            self.dict_stop_clusters[stop_id].append(new_guid)
        self.last_id_number +=1
        return new_guid

    def copy_cluster(self, cluster_ref_id):
        """
        Create the cluster red
        :param cluster_ref_id: the id of the reference cluster
        :return: the newly guid
        """
        new_guid = self._create_guid()
        cluster_ref = self[cluster_ref_id]

        new_cluster = cluster.Cluster.from_manager_stops(manager_stop=cluster_ref,
                                                         guid=new_guid,
                                                        tree=self.tree,
                                                         dict_cst= self.dict_cst,
                                                         list_useful_features=self.list_useful_features,
                                                         dict_disp=self.dict_dispersion)

        self[new_guid] = new_cluster
        new_cluster.tracking_evolution = cluster_ref.tracking_evolution.copy()
        assert not new_cluster.dict_features is None

        # Update the manager
        for stop_id in cluster_ref.keys():
            self.dict_stop_clusters[stop_id].append(new_guid)
        self.last_id_number +=1
        return new_guid

    def add_stop_to_cluster(self,stop,cluster_id):
        """
        add a stop to a cluster
        """
        self[cluster_id].add_stop(stop)
        self.dict_stop_clusters[stop.guid].append(cluster_id)

    def remove_stop_from_cluster(self,stop,cluster_id):
        """
        remove a stop from a cluster
        """
        self[cluster_id].remove_stop(stop)
        self.dict_stop_clusters[stop.guid].remove(cluster_id)


    def _create_guid(self):
        """
        Create a new guid for a clusters
        :return:
        """
        new_guid = 'clu_'+ str(self.last_id_number)
        assert not new_guid in self
        return new_guid


    def merge_cluster(self,guid_a, guid_b,manager_ref,iteration):
        """
        Merge two cluster to get a third one
        :param guid_a: the guid of the first cluster to merge
        :param guid_b: the guid of the second cluster to merge
        :param manager_ref: a manager ref for the stops
        :param iteration: the iteration at which the merge is performed
        :return: the new cluster_id
        """
        cluster_a = self[guid_a]
        cluster_b = self[guid_b]
        list_stop = set(list(cluster_a.keys())+ list(cluster_b.keys()))
        tracker = 'merging'
        new_guid = self.create_cluster(list_stop_id=list_stop, manager_ref=manager_ref,tracking = (tracker,iteration))
        return new_guid


    def check_cluster_initialized(self):
        """
        :return: a boolean indicating if all cluster have been correclty
        initalized
        """
        list_error = []
        for clu_id, clu in self.items():
            if clu.dict_features is None:
                list_error.append(clu_id)

        if len(list_error) >0:
            print('Clusters not init ',list_error)

        return not list_error

    def output_manager_clusters(self):
        """
        Prepare data to output the manager clusters
        :return: a list of dict,  ready to be written through pandas.
        """
        data = []
        for clu in self.values():
            data.append(clu.to_dict())

        return data


    def delete_cluster(self,clu_id):
        """
        Remove the cluster from the manager cluster
        :param clu_id: the id of the cluster
        Update everything
        """
        for stop_id in self[clu_id].keys():
            self.dict_stop_clusters[stop_id].remove(clu_id)

        del self[clu_id]

