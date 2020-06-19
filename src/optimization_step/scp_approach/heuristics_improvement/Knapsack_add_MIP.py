import gurobipy

class KnapsackFeaturesMIP(object):
    """
    Assume that each stop adds to the total features linearly, and therefore
    solve a Knapsack problems to maximize the utility
    """
    def __init__(self,dict_feat_cap,dict_stop_feat_weight,dict_stop_dual):
        self.dict_feat_cap = dict_feat_cap                      # dict[feat] = max_cap
        self.dict_stop_feat_weight = dict_stop_feat_weight      # dict[stop_id][feat] = weight
        self.dict_stop_dual = dict_stop_dual                    # dict[stop_id] = dual value

        # MIP parameters
        self.modelOptim = gurobipy.Model("MIP for set Knapscak problem add heuristic")
        self.modelOptim.Params.LogToConsole = 0
        self.modelOptim.modelSense = gurobipy.GRB.MAXIMIZE
        self.modelOptim.Params.LogFile = 'knapsak_mip.log'
        # self.modelOptim.Params.Method = -1

        self.EPSILON = 0.01

        # storage
        self.var_activation_stop = {}       # a dict[stop_id] = var


    def _create_var_stops(self):
        """
        Create the stop variable, binary
        """
        for stop_id in self.dict_stop_feat_weight:
            varname = 'act_' + stop_id
            revenue = self.dict_stop_dual[stop_id]
            self.var_activation_stop[stop_id] = self.modelOptim.addVar(0,1,revenue,gurobipy.GRB.BINARY,varname)


    def _cst_max_cap_feature(self):
        """
        Ensure that each stop doesn't push too much the boundaries so that we stay within the
        same leaf
        """
        for feature in self.dict_feat_cap:
            cst_name = 'Cap_' + feature
            cap = self.dict_feat_cap[feature]

            if cap >=0:
                self.modelOptim.addConstr(sum(self.var_activation_stop[stop_id] * self.dict_stop_feat_weight[stop_id][feature] for stop_id in self.var_activation_stop.keys()) <= cap,
                                          cst_name)
            else:
                self.modelOptim.addConstr(sum(self.var_activation_stop[stop_id] * self.dict_stop_feat_weight[stop_id][feature] for stop_id in self.var_activation_stop.keys()) >= cap,
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


    def solve(self):
        """
        Main function, solve the knapsack problem
        :return: the list of selected stop_id
        """
        self._create_var_stops()
        self._cst_max_cap_feature()
        self.modelOptim.optimize()

        return self._retrieve_solution()
