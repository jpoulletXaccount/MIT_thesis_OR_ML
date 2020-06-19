
import gurobipy
from src.learning_step.learning_trees.features_creation import derivingFeaturesCreatedDatasetLinear
from src.helpers import useful_name, useful_paths

class ClusteringCstParallelMIP(object):
    """
    Main class to derive the most relevant clusters of stop once we have the constraints of the trees
    """

    def __init__(self, dict_leaf_cst, dict_leaf_label,manager_stop):
        self.manager_stop = manager_stop
        self.dict_leaf_cst = dict_leaf_cst
        self.dict_leaf_label = dict_leaf_label
        assert len(self.dict_leaf_label) == len(self.dict_leaf_cst)

         # create object to derive features
        deriver_features = derivingFeaturesCreatedDatasetLinear.DerivingFeaturesCreatedDatasetLinear(manager_stop)
        self.dict_feature_stops = deriver_features.derive_features(for_mip=True)
        self.dict_dist_depot =deriver_features.distance_depot

        # Robustness
        self.ROBUSTNESS = 0.1   # percentage of safety we want

        # Pure optim
        self.modelOptim = gurobipy.Model("MIP cst Parallel absolute pixel relative features")
        self.modelOptim.modelSense = gurobipy.GRB.MINIMIZE
        #self.modelOptim.Params.tuneResults = 1
        #self.modelOptim.Params.Method = -1
        self.modelOptim.Params.Method = 1   # using dual simplex method (which has proved to be more performing)
        #self.modelOptim.Params.LogToConsole = 0
        self.EPSILON = 0.0001
        self.BIG_M = 10000


        # stocage
        self.var_activation_leaf = {}       # a dict[leaf_id] = var
        self.var_link_stop_leaf = {}        # a dict[leaf_id][stopId] = var
        self.var_feature_leaf = {}          # a dict[leaf_id] = dict of features variables (technically only intermediate variable)


    def _create_var_leaf(self):
        """
        Create all the variables corresponding to the leaf selected or nots
        :return:
        """
        for leaf_id in self.dict_leaf_label:
            varname = str(leaf_id) + "_ac"
            cost = self.dict_leaf_label[leaf_id]
            self.var_activation_leaf[leaf_id] = self.modelOptim.addVar(0,1,cost,gurobipy.GRB.BINARY,varname)


    def _create_var_stop_leaf(self):
        """
        Create all the variables corresponding to a stop being assigned to leaf or not
        :return:
        """
        list_stop_id = list(self.manager_stop.keys())

        for leaf_id in self.dict_leaf_label:
            self.var_link_stop_leaf[leaf_id] = {}

            for stop_id in list_stop_id:
                varname = str(leaf_id) + "_" + stop_id
                self.var_link_stop_leaf[leaf_id][stop_id] = self.modelOptim.addVar(0,1,0,gurobipy.GRB.BINARY,varname)


    def _create_var_feature_leaf(self):
        """
        Create all the variables corresponding to the required features.
        :return:
        """
        for leaf_id in self.dict_leaf_label:
            self.var_feature_leaf[leaf_id] = {}

            for feature_name in self.dict_feature_stops:
                varname = leaf_id +"_" + feature_name
                self.var_feature_leaf[leaf_id][feature_name] = self.modelOptim.addVar(0,gurobipy.GRB.INFINITY,0,gurobipy.GRB.CONTINUOUS,varname)


    def _cst_assignment_stop(self):
        """
        Ensure that each stop is assigned to exactly one cluster
        :return:
        """
        for stopId in self.manager_stop:
            cst_name = stopId + "_ass"

            self.modelOptim.addConstr(sum(self.var_link_stop_leaf[leaf_id][stopId] for leaf_id in self.dict_leaf_label) == 1,
                                      cst_name)


    def _cst_activation_leaf(self):
        """
        Ensure that a stop is assigned to a leaf iff the later is activated
        :return:
        """
        for leaf_id in self.dict_leaf_label:
            for stop_id in self.var_link_stop_leaf[leaf_id]:
                cst_name = "cst_act_" + leaf_id + stop_id

                self.modelOptim.addConstr(self.var_link_stop_leaf[leaf_id][stop_id] <= self.var_activation_leaf[leaf_id], cst_name)


    def _cst_feature_stop(self,feature_name):
        """
        Create every features related to stops (opposed to demand)
        :return:
        """
        list_considered = self.dict_feature_stops[feature_name]
        for leaf_id in self.dict_leaf_label:
            cst_name = leaf_id + "_" + feature_name
            self.modelOptim.addConstr(self.var_feature_leaf[leaf_id][feature_name] ==\
                                      sum(self.var_link_stop_leaf[leaf_id][stop_id] for stop_id in list_considered),
                                      cst_name)


    def _cst_feature_demand(self,feature_name):
        """
        Create every features related to demand (as opposed to stop)
        :return:
        """
        list_considered = self.dict_feature_stops[feature_name]
        for leaf_id in self.dict_leaf_label:
            cst_name = leaf_id + "_" + feature_name
            self.modelOptim.addConstr(self.var_feature_leaf[leaf_id][feature_name] ==\
                                      sum(self.var_link_stop_leaf[leaf_id][stop_id] * self.manager_stop[stop_id].demand for stop_id in list_considered),
                                      cst_name)

    def _cst_feature_distance(self,feature_name):
        """
        Create every features related to distance (as opposed to stop)
        :return:
        """
        list_considered = self.dict_feature_stops[feature_name]

        for leaf_id in self.dict_leaf_label:
            cst_name = leaf_id + "_" + feature_name
            self.modelOptim.addConstr(self.var_feature_leaf[leaf_id][feature_name] ==\
                                      sum(self.var_link_stop_leaf[leaf_id][stop_id] * self.dict_dist_depot[stop_id] for stop_id in list_considered),
                                      cst_name)

    def _cst_Kmean(self):
        """
        Ensure that two stops too further apart cannot be put in the same cluster
        :return:
        """
        for leaf_id in self.dict_leaf_label:
            for stop_id_1 in self.dict_pair_too_far:
                list_too_far = self.dict_pair_too_far[stop_id_1]

                for stop_id_2 in list_too_far:
                    cst_name = leaf_id +"_Kmean_" + stop_id_1 + "_" + stop_id_2
                    self.modelOptim.addConstr(self.var_link_stop_leaf[leaf_id][stop_id_1] + self.var_link_stop_leaf[leaf_id][stop_id_2] <=1,
                                              cst_name)


    def _cst_tree(self):
        """
        Create the cst of the tree such that we land on the right leaf
        :return:
        """
        for leaf_id in self.dict_leaf_cst:

            for i,cst in enumerate(self.dict_leaf_cst[leaf_id]):
                cst_name = leaf_id + "_cst_" + str(i)

                feat_name = cst.get_feature_name(self.dict_feature_stops)

                if cst.is_greater:
                    self.modelOptim.addConstr(self.var_feature_leaf[leaf_id][feat_name] >= (1 - self.ROBUSTNESS) * cst.right_side - self.BIG_M * (1 - self.var_activation_leaf[leaf_id]),
                                              cst_name)
                else:
                    self.modelOptim.addConstr(self.var_feature_leaf[leaf_id][feat_name] + self.EPSILON <= (1 - self.ROBUSTNESS) * cst.right_side + self.BIG_M * (1 - self.var_activation_leaf[leaf_id]),
                                              cst_name)


    def _initalize_model(self):
        """
        Create all variables and constraints
        :return:
        """
        # variables
        self._create_var_feature_leaf()
        self._create_var_leaf()
        self._create_var_stop_leaf()
        print("variables have been created ")

        # classical clustering cst
        self._cst_activation_leaf()
        self._cst_assignment_stop()

        # self._cst_Kmean()

        # features cst
        for feat_name in self.dict_feature_stops:
            if useful_name.DEMAND in feat_name:
                self._cst_feature_demand(feat_name)
            elif useful_name.DEPOT in feat_name:
                self._cst_feature_distance(feat_name)
            else:
                assert useful_name.DEPOT in feat_name or useful_name.NB_STOPS in feat_name,feat_name
                self._cst_feature_stop(feat_name)

        # tree constraints
        self._cst_tree()

        print("constraints have been created ")


    def _retrieve_solution(self):
        """
        Retrieve the clusters based on the results of the MIP
        :return: a dict[leaf_id] = list of stop id
        """
        dict_final_taken = {}
        for leaf_id in self.var_activation_leaf:
            if abs(self.var_activation_leaf[leaf_id].x -1) <= self.EPSILON:
                dict_final_taken[leaf_id] = []

                for stop_id in self.var_link_stop_leaf[leaf_id]:
                    if abs(self.var_link_stop_leaf[leaf_id][stop_id].x -1) <= self.EPSILON:
                        dict_final_taken[leaf_id].append(stop_id)

        assert len(self.manager_stop) == sum(len(lis) for lis in dict_final_taken.values()),  sum(len(lis) for lis in dict_final_taken.values())

        expected_number_vehi = sum(self.dict_leaf_label[leaf_id] for leaf_id in dict_final_taken)
        print("Based on the MIP, we are expecting ", expected_number_vehi, " vehicles for a number of clusters ", len(dict_final_taken))

        return dict_final_taken


    def solve(self):
        """
        Main function, create all variables and constraints, then solve the MIP and return the solution
        :return: a dict[leaf_id] = list of stop id, each corresponding to one cluster
        """
        self._initalize_model()
        self.modelOptim.write(useful_paths.PATH_TO_CONFIGURATION + 'saved_mip/model_cst_parallel.lp')
        self.modelOptim.optimize()
        self.modelOptim.write(useful_paths.PATH_TO_CONFIGURATION + 'saved_mip/model_cst_parallel.sol')
        return self._retrieve_solution()






