from src.data_set_creation.routing_model import routing_solver
from src.helpers import useful_paths,useful_name, local_path

import os
import pandas as pd

class RoutingSolverCVRPTW(routing_solver.RoutingSolver):
    """
    solver specific to CVRP
    """

    def __init__(self,manager_stop, config_cst,time_routing = 20):
        super(RoutingSolverCVRPTW,self).__init__(manager_stop,config_cst,time_routing)


    def solve_routing(self):
        """
        Main function, solve the routing and output the results in the corresponding file
        :return:
        """
        # write the input file for the solver
        input_file =open(useful_paths.FILE_TO_INPUT_SOLVER_CVRPTW, 'w')
        self.config_cst.dump_to_file(input_file)
        map_new_old_id = self.manager_stop.dump_to_file(input_file)
        input_file.close()
        self.execute_command_solver()
        self.store_results_in_csv(useful_paths.FILE_TO_STORAGE_CVRPTW,map_new_old_id)
        return self.parse_results(map_new_old_id)


    def solve_parse_routing(self):
        """
        Main function, solve the routing and return the results
        :return: a list of route
        """
        # write the input file for the solver
        input_file =open(useful_paths.FILE_TO_INPUT_SOLVER_CVRPTW, 'w')
        self.config_cst.dump_to_file(input_file)
        map_new_old_id = self.manager_stop.dump_to_file(input_file)
        input_file.close()
        self.execute_command_solver()
        return self.parse_results(map_new_old_id)


    def execute_command_solver(self):
        """
        Execute the command to solve the CVRP problem with the previously outputted problem
        :return:
        """
        if os.path.exists(useful_paths.FILE_TO_OUTPUT_SOLVER_CVRPTW):
            os.remove(useful_paths.FILE_TO_OUTPUT_SOLVER_CVRPTW)
        comp = 0
        while not os.path.exists(useful_paths.FILE_TO_OUTPUT_SOLVER_CVRPTW):
            l = os.getcwd()

            if l.split('/')[-1] == 'src' or l.split('\\')[-1] == 'src':
                path_use = os.path.join('..','..','solver','cvrptw','cvrptw.lsp')
                instance_name = os.path.join('..','..','solver','cvrptw','instances','input_cvrptw.txt')
                results_name = os.path.join('..','..','solver','cvrptw','results_routing.txt')
                output_name = os.path.join('..','..','solver','cvrptw','output.txt')
            else:
                path_use = os.path.join('..','..','..','..','solver','cvrptw','cvrptw.lsp')
                instance_name = os.path.join('..','..','..','..','solver','cvrptw','instances','input_cvrptw.txt')
                results_name = os.path.join('..','..','..','..','solver','cvrptw','results_routing.txt')
                output_name = os.path.join('..','..','..','..','solver','cvrptw','output.txt')


            myCmd = 'localsolver ' + path_use + '  inFileName=' +instance_name +' solFileName=' + results_name +' lsTimeLimit=' + str(self.time_routing) +' > ' + output_name

            #os.system('pwd')
            os.system(myCmd)

            comp +=1
            if comp >= 10:
                print("really weird, have been runnning more than 10 times")
                assert False

    @staticmethod
    def parse_results(map_new_old_id):
        """
        Read the routing results and return the correposnding results
        :param map_new_old_id: a dict[fileId] = stopId
        :return: a list of routes/vehicles
        """
        results_file = open(useful_paths.FILE_TO_OUTPUT_SOLVER_CVRPTW, 'r')
        line = results_file.readline()
        comp = 0
        list_routes = []
        num_vehicle = 0
        tot_distance = 0
        while line:
            words = line.strip().split(" ")
            if comp == 0:
                num_vehicle = int(words[0])
                tot_distance = float(words[1])
            else:
                list_stop_id = [map_new_old_id[int(words[i])].split("_")[1] for i in range(1,len(words))]  # Go back to the initial Id for the stop
                list_routes.append(list_stop_id)
            line = results_file.readline()
            comp +=1

        results_file.close()
        return num_vehicle,tot_distance,list_routes


    def store_results_in_csv(self,filename,map_new_old_id):
        """
        Store the routing results in filename, under the from of an exploitable file
        :param filename: the storage filename
        :param map_new_old_id: a dict[fileId] = stopId
        :return:
        """
        row = dict()
        row[useful_name.INPUT_FILE] = self.config_cst.file
        row[useful_name.CLUSTERING_METHOD] = self.config_cst.cluster_method
        row[useful_name.TIME_CST] = self.manager_stop.depot.due_date
        row[useful_name.CAPACITY_CST] = self.config_cst.capacity_cst

        results_file = open(useful_paths.FILE_TO_OUTPUT_SOLVER_CVRPTW, 'r')
        line = results_file.readline()
        comp = 0
        all_stop = set()
        dist_per_vehicle = []
        stop_per_vehicle = []
        while line:
            words = line.strip().split(" ")
            if comp == 0:
                row[useful_name.NUM_VEHICLE] = int(words[0])
                row[useful_name.TOTAL_DISTANCE] = words[1]
            else:
                set_stop_id = set([map_new_old_id[int(words[i])].split("_")[1] for i in range(1,len(words))])   # Go back to the initial Id for the stop
                all_stop = all_stop.union(set_stop_id)
                dist_per_vehicle.append(words[0])
                stop_per_vehicle.append(list(set_stop_id))
            line = results_file.readline()
            comp +=1

        results_file.close()
        assert row[useful_name.NUM_VEHICLE] == len(stop_per_vehicle), str(row[useful_name.NUM_VEHICLE]) +" " + str(len(stop_per_vehicle))

        row[useful_name.DIST_PER_VEHICLE] = self.convert_list_to_string(dist_per_vehicle)
        row[useful_name.STOPS_PER_VEHICLE] = self.conver_list_of_list_to_string(stop_per_vehicle)
        row[useful_name.ALL_STOPS_ID] = self.convert_list_to_string(list(all_stop))

        header = True
        if os.path.exists(filename):
            header = False
        data_results = [row]
        data_results = pd.DataFrame(data_results)
        data_results.to_csv(filename, header=header, index=False,mode = "a")

    @staticmethod
    def convert_list_to_string(list_to_convert):
        """
        Useful function to convert a list as a tring
        :return: string "elt0_elt1_...._elti"
        """
        final_string = ""
        for i,elt in enumerate(list_to_convert):
            if i < len(list_to_convert) -1:
                final_string += str(elt) +"_"
            else:
                final_string += str(elt)

        return final_string

    @staticmethod
    def conver_list_of_list_to_string(list_of_list):
        """
        Useful function to convert a list of list as string
        :param list_of_list: string(list)--string(list) .... --string(list)
        :return:
        """
        final_string = ""
        for i,elt in enumerate(list_of_list):
            if i < len(list_of_list) -1:
                final_string += RoutingSolverCVRPTW.convert_list_to_string(elt) +"--"
            else:
                final_string += RoutingSolverCVRPTW.convert_list_to_string(elt)

        return final_string


