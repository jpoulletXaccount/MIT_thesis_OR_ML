from src.data_set_creation.stops.objects import stops_cvrptw_ups


class Stop_ups_project(stops_cvrptw_ups.Stop_ups):
    """
    Class of stop corresponding to the capacitated vehicle routing problem with time window
    """
    def __init__(self,guid,x,y,demand,beginTW,endTW,stop_type,date,list_pack,initial_ups_id,slic):
        super(Stop_ups_project,self).__init__(guid,x,y,demand,beginTW,endTW,stop_type,date)
        self.initial_ups_id = initial_ups_id
        self.SLIC = slic
        self.beginTW = beginTW
        self.endTW = endTW             # we don't want it to be rescale to 0

        if self.stop_type == 1:
            self.demand = len(list_pack)
        assert self.demand == len(list_pack), print(self.demand, len(list_pack))
        self.list_pack = list_pack

    @property
    def volume(self):
        return sum(p.volume for p in self.list_pack)

    @property
    def weight(self):
        return sum(p.weight for p in self.list_pack)

