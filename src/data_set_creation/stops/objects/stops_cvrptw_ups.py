from src.data_set_creation.stops.objects import stops_cvrptw

from math import radians, cos, sin, asin, sqrt

class Stop_ups(stops_cvrptw.Stop_cvrptw):
    """
    Class of stop corresponding to the capacitated vehicle routing problem with time window
    """
    def __init__(self,guid,x,y,demand,beginTW,endTW,stop_type,date):
        if demand <= 15:
            service_time = 1.34 * demand
        else:
            service_time = 0.7 * 1.34 * demand

        # check pickup
        if stop_type == 1:
            demand = 0
            service_time = 5

        super(Stop_ups,self).__init__(guid,x,y,demand,max(0,beginTW-892),endTW-892,service_time)
        self.stop_type = stop_type
        self.date = date
        self.service_time = float(service_time)
        self.x = float(x)       # corresponds to lat
        self.y = float(y)       # corresponds to long


    def get_distance_to_another_stop(self,stop):
        """
        Compute the distance with an other stop
        :param stop: the other stop, assumed with lat long as well
        :return: the euclidiean distance, in kilometers
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [self.y, self.x, stop.y, stop.x])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers is 6371
        dist = 6371 * c
        return dist
