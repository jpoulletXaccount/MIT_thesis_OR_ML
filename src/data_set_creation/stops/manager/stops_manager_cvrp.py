from src.helpers import customs_exception
from src.data_set_creation.stops.objects import stops_cvrp
from src.data_set_creation.stops import classDepot
from src.data_set_creation.stops.manager import stops_manager

import os,json

class StopsManagerCRVP(stops_manager.StopsManager):
    """
    Inherits from dict, gather all orders at the same place
    """

    def __init__(self,depot = None):
        super(StopsManagerCRVP, self).__init__(depot)


    @classmethod
    def from_cvrp_file(cls,filename,check_dist_matrix):
        """
        Create a stop manager filled
        :param filename: the file from which we should read the stops
        :param check_dist_matrix: indicates if we should check for the dist matrix or not
        :return: an object of this class
        """
        manager = cls()
        need_dist_matrix = True

        # chek if need_distance_matrix
        dist_test = filename[0:-3]
        dist_test += 'json'
        if os.path.exists(dist_test) and check_dist_matrix:
            with open(dist_test, "r") as f:
                manager.matrix_stops_dist =json.loads(f.read())
            need_dist_matrix = False

        file = open(filename, 'r')  # reading only
        line = file.readline()
        reached_coord_section = False
        reached_demand_section = False
        reached_depot_section = False
        while line:
            words = line.strip().split(" ")
            # check if the line corresponds to the demand
            if reached_demand_section:
                if words[0] == 'DEPOT_SECTION':
                    reached_demand_section = False
                    reached_depot_section = True
                else:
                    manager._update_demand(line)
            # check if it corresponds to the coordinate
            elif reached_coord_section:
                if words[0] == 'DEMAND_SECTION':
                    reached_demand_section = True
                    reached_coord_section = False
                else:
                    manager._create_stop(line,need_dist_matrix)
            # check if corresponds to depot section
            elif reached_depot_section:
                manager._create_depot(line)
                reached_depot_section = False
            # check if next one is going to be
            else:
                if words[0] == "NODE_COORD_SECTION":
                    reached_coord_section = True
                elif words[0] == 'DEMAND_SECTION':
                    reached_demand_section = True

            line = file.readline()

        file.close()

        if check_dist_matrix and need_dist_matrix:
            json_file = json.dumps(manager.matrix_stops_dist)
            f = open(dist_test, "w")
            f.write(json_file)
            f.close()

        return manager

    def _create_stop(self,line,need_dist_matrix):
        """
        From the line of the file, create a stop with the corresponding
        :param line: a line from the file
        :return: a stop object
        """
        words = line.strip().split(" ")
        if len(words) != 3:
            raise customs_exception.WrongFile()

        guid = self._check_guid(words[0])
        stop = stops_cvrp.Stop_cvrp(guid, words[1], words[2], 0)

        # Update the distance matrix
        self.matrix_depot_dist[stop.guid] = stop.get_distance_to_another_stop(self.depot)
        if need_dist_matrix:
            self.matrix_stops_dist[stop.guid] = {}
            self.matrix_stops_dist[stop.guid][stop.guid] = 0
            for stop_id in self.keys():
                self.matrix_stops_dist[stop_id][stop.guid] = stop.get_distance_to_another_stop(self[stop_id])
                self.matrix_stops_dist[stop.guid][stop_id] = self[stop_id].get_distance_to_another_stop(stop)

        self[guid] = stop

    def _create_depot(self,line):
        """
        Create a depot
        :param line: the line correspondoing to the depot point
        :return:
        """
        words = line.strip().split(" ")
        if len(words) != 2:
            raise customs_exception.WrongFile()
        self.depot = classDepot.Depot(words[0], words[1])


    def _update_demand(self,line):
        """
        From the line of the file set the demand of the corresponding stop
        :param line: line from the file
        :return:
        """
        words = line.strip().split(" ")
        if len(words) != 2:
            raise customs_exception.WrongFile()

        guid = self.get_guid(words[0])
        self[guid].demand = int(words[1])


    def check_demand_updated(self):
        """
        Check that the demand is not null for any of the stops
        :return: a boolean
        """
        for stop_id in self.keys():
            if self[stop_id].demand == 0 and self[stop_id].stop_type ==2:
                return False

        return True

    def dump_to_file(self,file):
        """
        Write in the vrp format the manager stop
        :param file: the corresponding file
        :return: a map[id in file] = stopId
        """
        dimension_text = "DIMENSION : " + str(len(self) +1)
        file.write(dimension_text + "\n")
        map_new_old_id = self._dump_node_coord_section(file)
        self._dump_node_demand_section(file,map_new_old_id)
        self._dump_depot_section(file)
        return map_new_old_id


    def _dump_node_demand_section(self,file,map_new_old_id):
        """
        Write in the vrp format the demand section
        :param file: the corresponding file
        :param map_new_old_id: a dict[new_stop_id] = old_stop_id
        :return:
        """
        demand_text = "DEMAND_SECTION"
        file.write(demand_text + "\n")
        # First corresponds to the depot, with a demand of zero
        text_demand = str(1) + " " + str(0)
        file.write(text_demand +"\n")
        for newId in map_new_old_id.keys():
            stopId = map_new_old_id[newId]
            stop = self[stopId]
            text_demand = str(newId) + " " + str(stop.demand)
            file.write(text_demand +"\n")

    @property
    def demand(self):
        return sum(stop.demand for stop in self.values())










