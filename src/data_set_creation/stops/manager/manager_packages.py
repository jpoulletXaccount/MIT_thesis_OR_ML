
from scipy.stats import expon
import pandas as pd
import numpy as np
from src.data_set_creation.stops.objects.packages import Package

class ManagerPackages(dict):
    """
    Corresponds to the managers of all packages in the center.
    Is a dictonnary linking each package to its ID.
    """

    def __init__(self):
        super().__init__()  #corresponds to the initialisation of the dict class
        self.day = None

    @classmethod
    def from_package_ups_file(cls,date,filename):
        """
        Read the package file given and create the packages mentionned at the date
        :param date: the date we are interested in. If none then all of them
        :param filename: the corresponding filename
        :return:
        """
        manager = cls()
        # This just tells us if the df_date has already been prefiltered according to the configuration.
        df = pd.read_csv(filename,converters={'Date':int})
        if not date is None:
            df = df[df['Date'].isin(date)]

        for i,row in df.iterrows():
            manager._create_package(row)

        return manager

    def _create_package(self, row):
        """
        Loads the packages
        :param row: corresponds to the a package
        """
        IDpackage = row['IDpackage']
        self[IDpackage] = Package(
                                ID=IDpackage,
                                stopId= row['ID'],
                                weight = row['Weight'],
                                volume =  row['Volume']
                        )


    def get_pack_for_stop(self,stop_intial_id):
        """
        Get the list of package corresponding to the stop_id
        :param stop_intial_id: the stop_id
        :return: a list of packages objects
        """
        list_pack = []
        for pack in self.values():
            if pack.stopId == stop_intial_id:
                list_pack.append(pack)

        return list_pack


    @staticmethod
    def _randomWeightGeneration():
        """
        :return: A random weight based on the distribution defined below
        NOTE CURRENTLY NOT USED JUST 1
        """
        gamma = .12
        val =  expon.rvs(loc=1, scale=1 /gamma, size=1)
        return val[0]

    @staticmethod
    def _randomVolumeGeneration():
        """
        :return: A random Volume based on the distribution defined below
        """

        gamma = .0005
        val = expon.rvs(loc=1, scale=1 / gamma, size=1)
        return val[0]

    def getDictOfStopsID(self,listPackageID):
        """
        :param listPackageID: the list of package which we want to group by stop
        :return: a dict [stopId] = list package belonging to that stops
        """
        dictStopIdListPackage = {}

        for packageId in listPackageID:
            package = self[packageId]
            stopId = package.getStopId()

            if not stopId in dictStopIdListPackage.keys():
                dictStopIdListPackage[stopId] = [packageId]
            else:
                dictStopIdListPackage[stopId].append(packageId)

        assert (sum(len(dictStopIdListPackage[stopId]) for stopId in dictStopIdListPackage.keys()) == len(listPackageID))

        return dictStopIdListPackage

    def getAverageWeight(self):
        """
        :return: the average weight of all the packages in objects_manager
        """
        pkg_weights = list()
        for pkg_id in self:
            pkg_weights.append(self[pkg_id].getWeight())
        return np.mean(pkg_weights)

    def getAverageVolume(self):
        """
        :return: the average weight of all the packages in objects_manager
        """
        pkg_weights = list()
        for pkg_id in self:
            pkg_weights.append(self[pkg_id].getVolume())
        return np.mean(pkg_weights)

    # def getPkgType(self, pkg_id):
    #     """
    #     Returns the type of the package that has pkg_id
    #     :param pkg_id: The package ID of the package you are interested in
    #     :return: the type of package
    #     """
    #     # Volume in in^3, weight in lbs
    #     if self[pkg_id].getVolume() > 25**3 or self[pkg_id].getWeight() > 40:
    #         return pathParameters.PKG_TYPE_IRREG
    #     elif self[pkg_id].getVolume() > 11**3 or self[pkg_id].getWeight() > 11:
    #         return pathParameters.PKG_TYPE_BIG
    #     return pathParameters.PKG_TYPE_SMALL

    def getAllPackageFromStopType(self,stopType,managerStop):
        """
        Retrieve the list of packages corresponding to the stopType mentionned
        :return: a list of packId
        """
        listPack = []
        for packId in self:
            package = self[packId]
            stop = managerStop[package.stopId]
            if stop.stoptype == stopType:
                listPack.append(packId)

        return listPack
