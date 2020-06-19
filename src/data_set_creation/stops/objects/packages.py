


class Package(object):
    """
    Represents instances of UPS packages.
    """

    def __init__(self, ID, stopId, weight, volume):
        """
        :param ID: The unique ID of the package
        :param stopId: THe stop ID to which it contains. Should be Initial Stop ID
        :param weight: The weight of the package. [lbs]
        :param volume: The volume of the package. [in^3]
        """
        self.ID = ID
        self.stopId = stopId
        self.weight = weight    # Lbs
        self.volume = volume    # in^3

    def getStopId(self):
        """
        :return: The real stop ID to which the package pertains to
        """
        return self.stopId

    def getWeight(self):
        """
        :return: Return the weight of the package
        """
        return self.weight

    def getVolume(self):
        """
        :return: Return the weight of the package
        """
        return self.volume
