
import numpy as np

from src.learning_step.pixelation import managerPixel
from src.helpers import customs_exception
from src.learning_step.pixelation import pixel

class ManagerPixelRelative(managerPixel.ManagerPixel):
    """
    Manage the pixels, i.e. a collection of pixels
    """

    def __init__(self,depot):
        super(ManagerPixelRelative, self).__init__(depot)
        self.center = None
        self.len_pixel = None
        self.number_each_side = None


    def _create_pixel_far_away(self,min_x,max_x,min_y,max_y,id):
        """
        Create a pixel corresponding to the area
        :param min_x:
        :param max_x:
        :param min_y:
        :param max_y:
        """
        guid = "zz_far_" + str(id)      # zz because we are going to sort them by alphabetical order
        self[guid] = pixel.Pixel(guid, min_x, max_x, min_y, max_y)

    @classmethod
    def create_empty_pixel(cls,center_pixel, length, nb_pixels,depot):
        """
        Create a mnager of empty pixels
        :param center_pixel: the center of the pixels (x,y)
        :param length: the length of each pixel
        :param nb_pixels: the nb of pixels wanted. Note that we would need to square and then round that number
        :param depot: the depot we want to fit to the manager pixel
        :return: a pixel manager object
        """
        manager = cls(depot)
        manager.center = center_pixel
        manager.len_pixel = length

        center_x = center_pixel[0]
        center_y = center_pixel[1]

        nb_each_side = int(np.sqrt(nb_pixels))
        manager.number_each_side = nb_each_side

        begin_x = center_x - nb_each_side * length/2
        begin_y = center_y - nb_each_side * length/2

        # create the close pixels
        for i in range(0,nb_each_side):
            for j in range(0,nb_each_side):

                manager._create_pixel(min_x= begin_x + i *length,
                                      max_x= begin_x + (i+1) * length,
                                      min_y= begin_y + j *length,
                                      max_y= begin_y + (j+1) * length)

        # create the pixel far away
        infin = 100000
        comp = 0
        for k in [0,1]:
            for l in [0,1]:
                manager._create_pixel_far_away(min_x= center_x - (1-k) * infin,
                                               max_x= center_x + k * infin,
                                               min_y= center_y - (1-l) * infin,
                                               max_y= center_y + l * infin,
                                               id = comp)
                comp +=1

        assert comp ==4

        return manager


    @classmethod
    def create_from_manager_stop(cls,manager_stop,length, nb_pixels):
        """
        Create a pixel manager from a manager stop. I.e. create the pixels and assign some stops to them
        :param manager_stop: a manager with stops
        :param length: the length of each pixel
        :param nb_pixels: the nb of pixels wanted. Note that we would need to square and then round that number
        :return: a pixel manager object
        """
        center_pixel = manager_stop.centroid

        manager = cls(manager_stop.depot).create_empty_pixel(center_pixel, length, nb_pixels,manager_stop.depot)

        list_pixel = list(manager.keys())
        list_pixel.sort()   # make sure that the far away are last

        for stopId in manager_stop:
            stop = manager_stop[stopId]

            for pixelId in list_pixel:
                pixel_considered = manager[pixelId]

                if pixel_considered.does_contain_stop(stop):
                    break

            else:
                raise customs_exception.StopsDoesNotBelongToArea("This stop has not been found in the defined area " + str(stopId) + " " + str(stop.xy), " " + str(stop.y))

        assert manager._check_far()
        return manager


    def _check_far(self):
        """
        Check that the stop attributed to the far away are the only decision
        :return: a boolean
        """
        list_far = [k for k in self.keys() if "far" in k]

        x_min = self.center[0] - self.len_pixel * self.number_each_side /2
        x_max = self.center[0] + self.len_pixel * self.number_each_side /2
        y_min = self.center[1] - self.len_pixel * self.number_each_side /2
        y_max = self.center[1] + self.len_pixel * self.number_each_side /2

        for fa in list_far:
            pixel_far = self[fa]
            for stop in pixel_far.list_stop:
                if x_min <= stop.x < x_max and y_min <= stop.y < y_max:
                    print(stop.xy)
                    print(x_min,x_max)
                    print(y_min,y_max)
                    return False

        return True


