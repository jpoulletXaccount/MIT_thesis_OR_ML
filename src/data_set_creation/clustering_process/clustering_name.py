
import enum

class ClusteringMethod(enum.Enum):
    """
    Different types of methods for clustering orders
    """
    RANDOM = "RANDOM"
    KMEAN = "KMEAN"
    OPTIM = "OPTIM"
    NAIVE = "NAIVE"

    @classmethod
    def from_string(cls, value):
        """
        Try to convert value to ENUM type
        :param value: vlaue to convert
        :return:
        """
        key = str(value).upper()
        if key not in ClusteringMethod._member_map_:
            print(value,key,ClusteringMethod._member_map_)
            raise ValueError("Can't convert value")
        return ClusteringMethod._member_map_[key]

    
