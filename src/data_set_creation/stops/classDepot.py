from src.helpers import customs_exception

class Depot(object):
    """
    Depot: start and end of the vehicles
    """
    def __init__(self,x,y,max_time = None):
        if abs(int(x) - float(x)) <= 0.01:
            self.x = int(x)
            self.y = int(y)
        else:
            self.x = float(x)
            self.y = float(y)
        self._max_time = max_time

    @property
    def xy(self):
        return self.x,self.y

    @property
    def due_date(self):
        if self._max_time is None:
            raise customs_exception.NeedToBeInitialized
        return self._max_time

