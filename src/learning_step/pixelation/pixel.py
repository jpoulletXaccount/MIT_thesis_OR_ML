import numpy as np

class Pixel(object):
    """
    A pixel represents a certain area, such as a square defined by its lat/long
    """

    def __init__(self,guid,min_x,max_x,min_y,max_y):
        self.guid = guid
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.list_stop = []


    def does_contain_stop(self,stop):
        """
        Check if contains the stop, based on coordinates
        If yes, then add it to itself
        :param stop: a stop
        :return: a boolean
        """
        if self.min_x <= int(stop.x) < self.max_x:
            if self.min_y <= int(stop.y) < self.max_y:
                self.list_stop.append(stop)
                return True

        return False


    @property
    def demand(self):
        """
        Corresponds to the sum of the demand of the stops inside the pixel
        :return:
        """
        return sum(stop.demand for stop in self.list_stop)

    @property
    def xy(self):
        """
        :return: the geometric center of the pixel
        """
        center_x = (self.max_x + self.min_x) / 2
        center_y = (self.max_y + self.min_y) / 2
        return center_x,center_y

    @property
    def number_stop(self):
        return len(self.list_stop)

    @property
    def list_stop_guid(self):
        return [stop.guid for stop in self.list_stop]


    def number_stop_due_in_TW(self,tuple_TW):
        """
        Find all the stops due in the Time windows mentionned above
        :param tuple_TW: a tuple (beginTW, endTW)
        :return: such a number
        """
        total = 0
        for stop in self.list_stop:
            if tuple_TW[0]<= stop.endTW < tuple_TW[1]:
                total += 1

        return total

    def list_stop_due_in_TW(self,tuple_TW):
        """
        Find all the stops due in the Time windows mentionned above
        :param tuple_TW: a tuple (beginTW, endTW)
        :return: a list of these stops ID
        """
        total_list = []
        for stop in self.list_stop:
            if tuple_TW[0]<= stop.endTW < tuple_TW[1]:
                total_list.append(stop.guid)

        return total_list


    def get_distance_to_point(self,point):
        """
        Get the distance from the center of the pixel to the other point
        :param point: another point with coordinates
        :return: the dist
        """
        other_x,other_y = point.xy
        a= np.array(self.xy)
        b = np.array([other_x,other_y])
        return np.linalg.norm(a - b)


