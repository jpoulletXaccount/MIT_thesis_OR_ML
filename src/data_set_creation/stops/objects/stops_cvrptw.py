from src.data_set_creation.stops.objects import stops_cvrp


class Stop_cvrptw(stops_cvrp.Stop_cvrp):
    """
    Class of stop corresponding to the capacitated vehicle routing problem with time window
    """
    def __init__(self,guid,x,y,demand,beginTW,endTW,service_time):
        super(Stop_cvrptw,self).__init__(guid,x,y,demand)
        self.beginTW = int(beginTW)
        self.endTW = int(endTW)
        self.service_time = int(service_time)
        self.x = int(x)
        self.y = int(y)


    @property
    def TW(self):
        return self.beginTW,self.endTW

    def is_tight(self,inter):
        """
        :return: a boolean indicating if the time window interval is less than or not
        """
        return self.endTW - self.beginTW <= inter


