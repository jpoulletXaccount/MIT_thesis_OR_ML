
from src.data_set_creation.stops.manager import stops_manager_cvrptw_ups
from src.data_set_creation.stops.objects import stops_cvrptw_ups_project
from src.data_set_creation.stops import classDepot

import pandas as pd


class StopsManagerCVRPTW_UPS_Project(stops_manager_cvrptw_ups.StopsManagerCVRPTW_UPS):
    """
    Class of manager for stops_cvrptw
    """

    def __init__(self,depot = None):
        super(StopsManagerCVRPTW_UPS_Project,self).__init__(depot)


    @classmethod
    def from_ups_file_with_pack(cls,filename,date,manager_package):
        """
        Create a stop manager filled with packages
        :param filename: the file from which we should read the stops
        :param date: the date considered. If date is None then we consider all stops
        :param manager_package: a manager package
        :return: an object of this class
        """
        manager = cls()
        manager._create_depot(line=None)
        df = pd.read_csv(filename,converters={'Date':int})

        if not date is None:
            df = df[df['Date'].isin(date)]

        for i,row in df.iterrows():
            manager._create_stop_with_pack(row,manager_package)

        return manager


    @classmethod
    def from_sublist(cls,row_stops_pack,reference_manager_stop):
        """
        Create the manager stops corresponding to all stops listed in sublist
        :param row_stops_pack: the list of stops and pack to be put in the manager stops
        :param reference_manager_stop: the overall manager stops
        :return: an object manager stops
        """
        new_manager = cls(depot=reference_manager_stop.depot)
        new_manager.matrix_depot_dist = reference_manager_stop.matrix_depot_dist
        new_manager.matrix_stops_dist = reference_manager_stop.matrix_stops_dist

        list_stop_pack = row_stops_pack.split("|")
        for combo_stop_pack in list_stop_pack:
            stop_initial_id = combo_stop_pack.split(",")[0]
            if stop_initial_id != "V":
                all_pack_id = combo_stop_pack.split(",")[1:]

                ref_stop = reference_manager_stop[stop_initial_id]

                # Get the list of package
                list_pack = []
                for pack_id in all_pack_id:
                    list_pack.append(ref_stop.list_pack[int(pack_id)])

                new_guid = "sc_" + str(len(new_manager))
                assert not new_guid in new_manager, new_guid
                new_manager[new_guid] = stops_cvrptw_ups_project.Stop_ups_project(guid=new_guid,x=ref_stop.x,y=ref_stop.y,
                                                                                  demand=len(list_pack),beginTW=ref_stop.beginTW,endTW=ref_stop.endTW,
                                                                                  stop_type=ref_stop.stop_type,date=ref_stop.date,list_pack=list_pack,
                                                                                  initial_ups_id=stop_initial_id,slic=ref_stop.SLIC)

        return new_manager


    def _create_stop_with_pack(self,row,manager_pack):
        """
        From the line of the file, create a stop with the corresponding packages
        :param row: a row of the df
        :param manager_pack: a manager package
        :return:
        """
        guid = row['ID']
        assert not guid in self.keys(),guid
        # Get back the packages
        list_pack = manager_pack.get_pack_for_stop(guid)
        assert len(list_pack) == row['Npackage']
        stop = stops_cvrptw_ups_project.Stop_ups_project(guid, row['Latitude'], row['Longitude'], row['Npackage'], row['Start Commit Time'],row['End Commit Time'], row['Stop type'],row['Date'],
                                                               list_pack=list_pack,initial_ups_id=guid,slic=row['SLIC'])

        # Update the distance matrix
        self.matrix_depot_dist[stop.guid] = stop.get_distance_to_another_stop(self.depot)
        self.matrix_stops_dist[stop.guid] = {}
        self.matrix_stops_dist[stop.guid][stop.guid] = 0
        for stop_id in self.keys():
            self.matrix_stops_dist[stop_id][stop.guid] = stop.get_distance_to_another_stop(self[stop_id])
            self.matrix_stops_dist[stop.guid][stop_id] = self[stop_id].get_distance_to_another_stop(stop)

        self[guid] = stop

    def _create_depot(self,line):
        """
        From the line of the file create the corresponding depot
        :return:
        """
        self.depot = classDepot.Depot(42.4593, -73.1903, 892 + 1200)  # we set up a shift of 12h











