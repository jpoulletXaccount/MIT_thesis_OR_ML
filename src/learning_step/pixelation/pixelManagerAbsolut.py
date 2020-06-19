
import numpy as np

from src.learning_step.pixelation import managerPixel
from src.helpers import customs_exception

class ManagerPixelAbsolut(managerPixel.ManagerPixel):
    """
    Manage the pixels, i.e. a collection of pixels
    """

    def __init__(self,depot):
        super(ManagerPixelAbsolut, self).__init__(depot)


    @classmethod
    def create_empty_pixel(cls, dim_pixel, nb_pixels,depot):
        """
        Create a mnager of empty pixels
        :param dim_pixel: the dim (i.e [(min_x, max_x),(min_y,max_y)]
        :param nb_pixels: the nb of pixels wanted. Note that we would need to square and then round that number
        :param depot: the depot we want to fit to the manager pixel
        :return: a pixel manager object
        """
        manager = cls(depot)
        min_x = dim_pixel[0][0]
        max_x = dim_pixel[0][1]
        min_y = dim_pixel[1][0]
        max_y = dim_pixel[1][1]

        x_width = max_x - min_x + 1
        y_width = max_y - min_y + 1

        nb_each_side = int(np.sqrt(nb_pixels))

        x_adv = x_width/nb_each_side
        y_adv = y_width/nb_each_side

        for i in range(0,nb_each_side):
            for j in range(0,nb_each_side):
                manager._create_pixel(min_x=min_x + i*x_adv,
                                      max_x= min_x + (i+1) * x_adv,
                                      min_y= min_y + j *y_adv,
                                      max_y= min_y + (j +1) * y_adv)

        return manager


    @classmethod
    def create_from_manager_stop(cls,manager_stop,dim_pixel,nb_pixels):
        """
        Create a pixel manager from a manager stop. I.e. create the pixels and assign some stops to them
        :param manager_stop: a manager with stops
        :param dim_pixel: the dim (i.e [(min_x, max_x),(min_y,max_y)]
        :param nb_pixels: the nb of pixels wanted. Note that we would need to square and then round that number
        :return: a pixel manager object
        """
        manager = cls(manager_stop.depot).create_empty_pixel(dim_pixel, nb_pixels,manager_stop.depot)

        for stopId in manager_stop:
            stop = manager_stop[stopId]

            for pixelId in manager:
                pixel_considered = manager[pixelId]

                if pixel_considered.does_contain_stop(stop):
                    break

            else:
                raise customs_exception.StopsDoesNotBelongToArea("This stop has not been found in the defined area " + str(stopId) + " " + str(stop.xy), " " + str(stop.y))

        return manager


