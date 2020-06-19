class RedundantGuid(Exception):
    """Error raised when a guid is already used for an object"""
    pass

class WrongFile(Exception):
    """Error raised when the file structure does not correspond to what was expected"""
    pass

class FunctionNeedBeOverwritten(Exception):
    """Error raised when a function which should have been overwritten is not"""
    pass

class FunctionalityNotImplemented(Exception):
    """Error raised when a functionality has not been implemented yet"""
    pass


class NeedToBeInitialized(Exception):
    """Error raised when a value has not been initialized yet and is called"""

class StopsDoesNotBelongToArea(Exception):
    """Error raised when a stop has not been assigned to a pixel"""
