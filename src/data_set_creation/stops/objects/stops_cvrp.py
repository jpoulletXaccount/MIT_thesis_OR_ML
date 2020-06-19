from src.data_set_creation.stops.objects import stops


class Stop_cvrp(stops.Stop):
    """
    Class of stops corresponding to the capacitated vehicle routing problem
    """

    def __init__(self,guid,x,y,demand):
        super(Stop_cvrp,self).__init__(guid,x,y)
        self._demand = int(demand)

    @property
    def demand(self):
        return self._demand

    @demand.setter
    def demand(self,dem):
        assert dem >0
        self._demand = dem
