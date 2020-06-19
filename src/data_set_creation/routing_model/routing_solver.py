from src.helpers import customs_exception

class RoutingSolver(object):
    """
    Mother class of routing solver
    """
    def __init__(self, manager_stop, config_cst,time_routing):
        self.manager_stop = manager_stop
        self.config_cst = config_cst

        self.time_routing = time_routing


    def _dump_manager_stop(self,file):
        """
        Write the infomration contained in the manager stop to the file, in the correct format
        :param file: the corresponding file to which to dump
        :return:
        """
        self.manager_stop.dump_to_file(file)

    def _dump_config_cst(self,file):
        """
        Write the information contained in the config cst, in the correct format
        :param file: the corresponding file to which to dump
        :return:
        """
        self.config_cst.dump_to_file(file)


    def solve_routing(self):
        """
        Main function, should be overwritten
        :return:
        """
        raise customs_exception.FunctionNeedBeOverwritten()
