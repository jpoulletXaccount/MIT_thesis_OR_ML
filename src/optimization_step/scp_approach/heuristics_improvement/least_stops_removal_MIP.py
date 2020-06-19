import gurobipy

class LeastStopsRemovalMIP(object):
    """
    Assume that each stop adds to the total features linearly, and therefore
    solve a Knapsack problems to maximize the utility
    """
    def __init__(self, dict_feat_gap, dict_stop_feat_weight,dict_stop_dual_val):
        self.dict_feat_gap = dict_feat_gap                      # dict[feat] = gap
        self.dict_stop_feat_weight = dict_stop_feat_weight      # dict[stop_id][feat] = weight removal
        self.dict_stop_dual_val = dict_stop_dual_val            # dict[stop_id] = dual value

        # MIP parameters
        self.modelOptim = gurobipy.Model("MIP for removal heuristic")
        self.modelOptim.Params.LogToConsole = 0
        self.modelOptim.modelSense = gurobipy.GRB.MINIMIZE
        self.modelOptim.Params.LogFile = 'least_stop_mip.log'
        # self.modelOptim.Params.Method = -1

        self.EPSILON = 0.001
        self.bigM = 10000
        self.alpha = 10      # importance given to the reduced cost

        # storage
        self.var_activation_stop = {}       # a dict[stop_id] = var
        self.var_violation_cst = {}         # a dict[feature] = var


    def _create_var_stops(self):
        """
        Create the stop variable, binary
        """
        for stop_id in self.dict_stop_feat_weight:
            varname = 'act_' + stop_id
            cost = 1 + self.alpha * self.dict_stop_dual_val[stop_id]
            self.var_activation_stop[stop_id] = self.modelOptim.addVar(0,1,cost,gurobipy.GRB.BINARY,varname)


    def _create_var_violation(self):
        """
        Create the violation variable, binary
        """
        for feat in self.dict_feat_gap:
            varname = 'violation_' + feat
            cost = 0
            self.var_violation_cst[feat] = self.modelOptim.addVar(0,1,cost,gurobipy.GRB.BINARY,varname)


    def _cst_violation_feature(self,percentage =1):
        """
        Ensure that the violation value only takes value one if the features constraints are actually violated.
        :param percentage: the percentage of the gap to be considered
        """
        for feature in self.dict_feat_gap:
            cst_name = 'Violation_' + feature
            gap = percentage * self.dict_feat_gap[feature]

            if gap >=0:
                self.modelOptim.addConstr(sum(self.var_activation_stop[stop_id] * self.dict_stop_feat_weight[stop_id][feature] for stop_id in self.var_activation_stop.keys()) - gap >=
                                          - (1- self.var_violation_cst[feature]) * self.bigM,
                                          cst_name)
            else:
                self.modelOptim.addConstr(sum(self.var_activation_stop[stop_id] * self.dict_stop_feat_weight[stop_id][feature] for stop_id in self.var_activation_stop.keys()) - gap + self.EPSILON <=
                                          (1- self.var_violation_cst[feature]) * self.bigM,
                                          cst_name)


    def _cst_at_least_one_violated(self):
        """
        Make sure that at least one of the constraints is violated
        """
        cst_name = "at_least_one_violated"
        self.modelOptim.addConstr(sum(self.var_violation_cst[feat] for feat in self.var_violation_cst.keys()) >= 1,
                                  cst_name)

    def _cst_at_least_one_stop_selected(self):
        """
        Make sure that at least one of the stops is selected. Which may not necessarily be the case if we have to introduce
        a gap due to initial infeasibility
        """
        cst_name = "at_least_one_stop"
        self.modelOptim.addConstr(sum(self.var_activation_stop[stop_id] for stop_id in self.var_activation_stop.keys()) >= 1,
                                  cst_name)


    def _retrieve_solution(self):
        """
        :return: a list of selected_stop
        """
        list_stop_id = []
        for stop_id in self.var_activation_stop:
            var = self.var_activation_stop[stop_id]

            if abs(var.x) >= self.EPSILON:
                list_stop_id.append(stop_id)

        return list_stop_id


    def _deal_with_infeasible(self):
        """
        Deal with infeasibility problem which may arise due to the limited number of stops tested
        :return the list of selected stop_id or all_stops_id
        """
        nb_iter = 0
        while self.modelOptim.Status == gurobipy.GRB.INFEASIBLE and nb_iter < 7:
            nb_iter += 3
            # remove all constraints
            self.modelOptim.remove(self.modelOptim.getConstrs())
            self.modelOptim.update()

            # re -add the constraint
            self._cst_at_least_one_violated()
            percentage = (10-nb_iter)/10
            self._cst_violation_feature(percentage)
            self.modelOptim.optimize()


        if self.modelOptim.Status == gurobipy.GRB.INFEASIBLE:
            return list(self.var_activation_stop.keys())
        else:
            return self._retrieve_solution()


    def solve(self):
        """
        Main function, solve the knapsack problem
        :return: the list of selected stop_id
        """
        self._create_var_stops()
        self._create_var_violation()
        self._cst_at_least_one_violated()
        self._cst_violation_feature()
        self._cst_at_least_one_stop_selected()

        self.modelOptim.optimize()

        if self.modelOptim.Status == gurobipy.GRB.INFEASIBLE:
            return self._deal_with_infeasible()

        else:
            return self._retrieve_solution()
