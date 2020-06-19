import numpy as np

from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.data_set_creation.stops.objects import stops_cvrptw
from src.data_set_creation.stops import classDepot
from src.data_set_creation.config import config_constraint_cvrptw

class InstanceCreator(object):

    def __init__(self,output_file, cap, number_customer):
        self.output_file = output_file
        self.cap = cap
        self.number_cust = number_customer

        # Parmeters
        self.SIZE = 300
        self.PERCENTAGE_EARLY = 0.3
        self.PERCENTAGE_LATE = 0.3


    def _generate_manager(self):
        """
        Generate stop manager
        :return: a manager stop
        """
        depot = classDepot.Depot(x= self.SIZE/2, y = self.SIZE/2, max_time=1500)

        mana = stops_manager_cvrptw.StopsManagerCVRPTW(depot=depot)

        for i in range(0,self.number_cust):
            x = np.random.randint(0,self.SIZE)
            y = np.random.randint(0,self.SIZE)

            d = np.random.randint(1,10)

            a = np.random.uniform(0,1)
            if a < self.PERCENTAGE_EARLY:
                btw = 0
                etw = 250

            elif a > 1- self.PERCENTAGE_LATE:
                btw = 750
                etw = 1050

            else:
                btw = 0
                etw = 1050

            stop = stops_cvrptw.Stop_cvrptw(guid="stop_" + str(i),
                                            x=x,
                                            y=y,
                                            demand=d,
                                            beginTW=btw,
                                            endTW=etw,
                                            service_time=2*d)
            mana[stop.guid] = stop

        return mana


    def _generate_config(self):
        """
        Generate config
        :return: a config
        """
        config_cv = config_constraint_cvrptw.ConfigCVRPTW(cap=self.cap,
                                                          filename="created",
                                                          nb_vehi=100)
        return config_cv

    def generate_instance(self):
        """
        Main function, generate instances and ouput them
        :return:
        """
        config = self._generate_config()
        manager_stop = self._generate_manager()

        output_file = open(self.output_file,'w')
        text = "VEHICLE"
        output_file.write(text + "\n")
        text = "NUMBER   CAPACTIY"
        output_file.write(text + "\n")
        output_file.write("150   "+ str(config.capacity_cst) + "\n")


        # config.dump_to_file_capacity(output_file)
        manager_stop.dump_to_file(output_file)


