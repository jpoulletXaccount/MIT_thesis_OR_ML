
import gurobipy

class MIP_set_covering(object):
    """
    Solve a really classical set covering problem
    """
    def __init__(self,list_stop, dict_stop_clus, dict_clus_predict):
        self.list_stop = list_stop
        self.dict_stop_clus = dict_stop_clus
        self.dict_clus_prediction = dict_clus_predict

        # Pure optim
        self.modelOptim = gurobipy.Model("MIP for set covering problem")
        self.modelOptim.Params.LogToConsole = 0
        self.modelOptim.modelSense = gurobipy.GRB.MINIMIZE
        self.modelOptim.Params.TimeLimit = 1000
        self.modelOptim.Params.LogFile = 'set_covering_problem.log'
        # self.modelOptim.Params.Method = -1

        self.EPSILON = 0.01


        # storage
        self.var_activation_clu = {}       # a dict[cluster_id] = var
        self.cst_stops = {}                # a dict[stop_id] = cst


    def _create_var_cluster(self,relax):
        """
        Create var indicating if cluster have been chosen or not
        :param relax: boolean, indicates if we solve the relaxation or not.
        :return:
        """
        for clus_id in self.dict_clus_prediction:
            varname = "act_" + clus_id
            cost = self.dict_clus_prediction[clus_id]
            if relax:
                self.var_activation_clu[clus_id] = self.modelOptim.addVar(0,1,cost,gurobipy.GRB.CONTINUOUS,varname)
            else:
                self.var_activation_clu[clus_id] = self.modelOptim.addVar(0,1,cost,gurobipy.GRB.BINARY,varname)


    def _cst_covertness_stop(self):
        """
        Ensures that each stop is covered by at least one cluster
        :return:
        """
        for stop_id in self.list_stop:
            list_potential_cluster = self.dict_stop_clus[stop_id]
            cst_name = "covertness_" + stop_id
            cst = self.modelOptim.addConstr(sum(self.var_activation_clu[clus_id] for clus_id in list_potential_cluster) >=1,
                                      cst_name)
            self.cst_stops[stop_id] = cst


    def _init_model(self, relax):
        """
        Create all variables and constraints
        :param relax: boolean indicating if we solve the relaxation or not
        :return:
        """
        self._create_var_cluster(relax)
        self._cst_covertness_stop()

    def _set_up_warm_start(self,list_selected):
        """
        As a warm start, consider all clusters in the list selected
        :param list_selected: a list of cluster id
        """
        for clu_id in self.var_activation_clu:
            if clu_id in list_selected:
                self.var_activation_clu[clu_id].Start = 1
            else:
                self.var_activation_clu[clu_id].Start = 0

    def _retrieve_solution(self,relax):
        """
        :return: a list of cluster_id
        """
        list_cluster = []
        for clu_id in self.var_activation_clu:
            var = self.var_activation_clu[clu_id]

            if relax:
                if 0.9 - var.x <= self.EPSILON:
                    list_cluster.append(clu_id)
            else:
                if abs(var.x -1 ) <= self.EPSILON:
                    list_cluster.append(clu_id)

        return list_cluster


    def solve(self, relax,warm_start):
        """
        Main function, solve the set covering approach
        :param relax: boolean, indicates if we should solve the relaxation or not
        :param warm_start: if not None, indicates the clusters selected for the warm start
        :return: the list of selected cluster_id, a dict of reduced cost (cf clusters variable) and of dual price for
        all the stops to be covered (cf cst)
        """
        self._init_model(relax)

        if not warm_start is None:
            self._set_up_warm_start(warm_start)

        self.modelOptim.optimize()
        if self.modelOptim.Status == gurobipy.GRB.INFEASIBLE:
            print("Infeasible in set covering problem")
            self.modelOptim.computeIIS()
            self.modelOptim.write("scp.ilp")
            self.modelOptim.write("scp.mps")
            assert False

        list_selected_clusters = self._retrieve_solution(relax)
        if relax:
            dict_reduced_cost = {key_id : clu_var.RC for key_id, clu_var in self.var_activation_clu.items()}
            dict_dual_val = {stop_id : cst.Pi for stop_id, cst in self.cst_stops.items()}
            dict_x_value = {key_id : clu_var.x for key_id,clu_var in self.var_activation_clu.items()}
        else:
            dict_reduced_cost = {}
            dict_dual_val = {}
            dict_x_value = {}

        obj_val = self.modelOptim.getObjective().getValue()
        return list_selected_clusters, dict_reduced_cost, dict_dual_val,obj_val, dict_x_value




