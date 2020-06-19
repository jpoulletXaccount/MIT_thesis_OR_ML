import gurobipy

class MIP_set_covering(object):
    """
    Solve a really classical set covering problem
    """
    def __init__(self,list_stop,dict_clus_stop, dict_clus_predict):
        self.list_stop = list_stop
        self.dict_stop_clus = self._compute_stop_dict(dict_clus_stop)
        self.dict_clus_prediction = dict_clus_predict


        # Pure optim
        self.modelOptim = gurobipy.Model("MIP cst Parallel absolute pixel relative features")
        self.modelOptim.modelSense = gurobipy.GRB.MINIMIZE
        # self.modelOptim.Params.Method = -1

        self.modelOptim.Params.LogToConsole = 1
        self.EPSILON = 0.0001


        # stocage
        self.var_activation_clu = {}       # a dict[cluster_id] = var


    def _compute_stop_dict(self,dict_clus_stop):
        """
        :param dict_clus_stop: a dict[clus_id] = list stop id
        :return: a dict[stop_id] = list clus_id
        """
        # init
        dict_stop_clus = {}
        for stop_id in self.list_stop:
            dict_stop_clus[stop_id] = []

        for clus_id in dict_clus_stop:
            for stop_id in dict_clus_stop[clus_id]:
                dict_stop_clus[stop_id].append(clus_id)

        # check
        for stop_id in dict_stop_clus:
            assert len(dict_stop_clus[stop_id]) >=1, stop_id

        return dict_stop_clus


    def _create_var_cluster(self):
        """
        Create var indicating if cluster have been chosen or not
        :return:
        """
        for clus_id in self.dict_clus_prediction:
            varname = "act_" + clus_id
            cost = self.dict_clus_prediction[clus_id]
            self.var_activation_clu[clus_id] = self.modelOptim.addVar(0,1,cost,gurobipy.GRB.BINARY,varname)


    def _cst_covertness_stop(self):
        """
        Ensures that each stop is covered by at least one cluster
        :return:
        """
        for stop_id in self.list_stop:
            list_potential_cluster = self.dict_stop_clus[stop_id]
            cst_name = "covertness_" + stop_id
            self.modelOptim.addConstr(sum(self.var_activation_clu[clus_id] for clus_id in list_potential_cluster) >=1,
                                      cst_name)


    def _init_model(self):
        """
        Create all variables and constraints
        :return:
        """
        self._create_var_cluster()
        self._cst_covertness_stop()


    def _retrieve_solution(self):
        """
        :return: a list of cluster_id
        """
        list_cluster = []
        for clu_id in self.var_activation_clu:
            var = self.var_activation_clu[clu_id]

            if abs(var.x -1) <= self.EPSILON:
                list_cluster.append(clu_id)

        return list_cluster


    def solve(self):
        """
        Main function, solve the set covering approach
        :return:
        """
        self._init_model()
        self.modelOptim.optimize()
        return self._retrieve_solution()




