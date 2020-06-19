from src.data_set_creation.config import config_constraint
from src.helpers import customs_exception

class ConfigCVRPTW(config_constraint.Config):
    """
    Object which represents the configuration for the routing phase
    """

    def __init__(self,cap,filename,nb_vehi):
        super(ConfigCVRPTW, self).__init__(cap)
        self.time_cst = None
        self._origin_filename = filename
        self._number_vehicles = nb_vehi

    @classmethod
    def from_cvrptw_file(cls,filename):
        """
        Build a configuration corresponding to the cvrp problem
        :param filename: the filename corresponding
        :return: a ConfigCst object
        """
        file = open(filename, 'r')  # reading only
        line = file.readline()
        has_reached_vehicle_section = False
        cap = -1
        number = -1
        while line:
            words = line.strip().split("\t")
            if words[0] == 'VEHICLE':
                has_reached_vehicle_section = True
                file.readline()
                line = file.readline()
                words = line.strip().split(" ")
                words = [wo for wo in words if wo != '']
                number = int(words[0])
                cap = int(words[1])
                break
            line = file.readline()

        if not has_reached_vehicle_section:
            raise customs_exception.WrongFile()

        return cls(cap,filename,number)

    def to_print(self):
        return "cap: " + str(self.capacity_cst) + " and number vehicle: " + str(self._number_vehicles)


    def dump_to_file(self,file):
        """
        Write the constraint in the correct format
        :param file: the corresponding file to write
        :return:
        """
        words = self._origin_filename.split("/")
        file.write(words[-1] +"\n")
        file.write("\n")
        file.write("VEHICLE\n")
        file.write("NUMBER \t CAPACITY\n")
        file.write(str(self._number_vehicles) + "\t" + str(self.capacity_cst) + "\n")
        file.write("\n")


