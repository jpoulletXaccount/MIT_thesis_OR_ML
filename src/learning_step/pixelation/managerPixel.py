
from src.learning_step.pixelation import pixel

class ManagerPixel(dict):
    """
    Manage the pixels, i.e. a collection of pixels
    """

    def __init__(self,depot):
        super(ManagerPixel, self).__init__()
        self.depot = depot


    def _create_pixel(self,min_x,max_x,min_y,max_y):
        """
        Create a pixel corresponding to the area
        :param min_x:
        :param max_x:
        :param min_y:
        :param max_y:
        """
        assert min_x < max_x
        assert min_y < max_y
        guid = self._create_pixel_id()
        self[guid] = pixel.Pixel(guid, min_x, max_x, min_y, max_y)


    def _create_pixel_id(self):
        """
        Create the corresponding pixel ID
        :return: a string corresponding to the ID
        """
        string_id = "pixel_" + str(len(self))
        assert not string_id in self
        return string_id


    @property
    def number_stops(self):
        """
        :return: the number of stops contained in all pixels
        """
        return sum(pix.number_stop for pix in self.values())

    @property
    def total_demand(self):
        """
        :return: the total demand of the stops contained in the pixel
        """
        return sum(pix.demand for pix in self.values())


    @property
    def list_stop_id(self):
        """
        :return: a list of stop id contain in the pixels
        """
        list_id = []
        for pix in self.values():
            list_id.extend([stop.guid for stop in pix.list_stop])

        return list_id
