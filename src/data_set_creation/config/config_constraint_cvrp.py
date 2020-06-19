from src.data_set_creation.config import config_constraint
from src.helpers import customs_exception

class ConfigCVRP(config_constraint.Config):
    """
    Object which represents the configuration for the routing phase
    """

    def __init__(self,cap,dist,filename):
        super(ConfigCVRP, self).__init__(cap)
        self.distance_cst = dist
        self._origin_filename = filename

    @classmethod
    def from_cvrp_file(cls,filename):
        """
        Build a configuration corresponding to the cvrp problem
        :param filename: the filename corresponding
        :return: a ConfigCst object
        """
        file = open(filename, 'r')  # reading only
        line = file.readline()
        cap = -1
        dist = -1
        while line:
            words = line.strip().split(" ")
            if words[0] == 'CAPACITY:':
                cap = int(words[1])
            elif words[0] == 'DISTANCE:':
                dist = int(words[1])
            line = file.readline()

        if cap == -1 or dist == -1:
            print(cap,dist)
            raise customs_exception.WrongFile()

        return cls(cap,dist,filename)

    def to_print(self):
        return "cap: " + str(self.capacity_cst) + " and dist: " + str(self.distance_cst)

    def dump_to_file_distance(self,file):
        """
        Write the distance constraint to the file in the correct format
        :param file: the corresponding file to output
        :return:
        """
        dist_text = "DISTANCE : " + str(self.distance_cst)
        file.write(dist_text + '\n')


    def dump_to_file(self,file):
        """
        Write the constraint in the correct format
        :param file: the corresponding file to write
        :return:
        """
        self.dump_to_file_capacity(file)
        self.dump_to_file_distance(file)


