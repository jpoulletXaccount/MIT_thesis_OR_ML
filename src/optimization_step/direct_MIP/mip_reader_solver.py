
import gurobipy,os, sys

sys.path.append(os.getcwd().split("src")[0])

from src.helpers import useful_paths

def solve():
    """
    solve the model store under the file
    :return:
    """
    model = gurobipy.read(useful_paths.PATH_TO_CONFIGURATION + 'saved_mip/model_cst_parallel.lp')
    model.optimize()
    model.write(useful_paths.PATH_TO_CONFIGURATION + 'saved_mip/model_cst_parallel.sol')

def read_solution():
    """
    read the solution from a file
    :return:
    """
    model = gurobipy.read(useful_paths.PATH_TO_CONFIGURATION + 'saved_mip/model_cst_parallel.sol')


if __name__ == '__main__':
    read_solution()

