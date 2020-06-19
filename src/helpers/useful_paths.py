from src.helpers import local_path
import os

# General paths
PATH_TO_THESIS = local_path.PATH_TO_PROJECT
PATH_TO_CODE = PATH_TO_THESIS +'thesis/'
PATH_TO_CONFIGURATION = PATH_TO_CODE + 'configuration/'

# Benchmarks
PATH_TO_BENCHMARK = PATH_TO_THESIS + 'benchmarks/'
PATH_TO_BENCHMARK_LI = PATH_TO_BENCHMARK + 'cvrp/Li_benchmarks/'
PATH_TO_BENCHMARK_GOLDEN = PATH_TO_BENCHMARK + 'cvrp/Golden/'
PATH_TO_BENCHMARK_KYTOJOKI = PATH_TO_BENCHMARK + 'cvrp/Kytojoki_benchmarks/'
PATH_TO_BENCHMARK_SALOMON = PATH_TO_BENCHMARK + 'cvrptw/salomon/100_customer/'
PATH_TO_BENCHMARK_HOMBERGER_600 = PATH_TO_BENCHMARK + 'cvrptw/homberger/homberger_600_customer_instances/'
PATH_TO_BENCHMARK_HOMBERGER_800 = PATH_TO_BENCHMARK + 'cvrptw/homberger/homberger_800_customer_instances/'
PATH_TO_BENCHMARK_HOMBERGER_1000 = PATH_TO_BENCHMARK + 'cvrptw/homberger/homberger_1000_customer_instances/'
PATH_TO_BENCHMARK_CREATED = PATH_TO_BENCHMARK + 'cvrptw/created_new/'
PATH_TO_BENCHMARK_CREATED_600 = PATH_TO_BENCHMARK_CREATED + '600_customers/'
PATH_TO_BENCHMARK_CREATED_1000 = PATH_TO_BENCHMARK_CREATED + '1000_customers/'
PATH_TO_BENCHMARK_CREATED_1500 = PATH_TO_BENCHMARK_CREATED + '1500_customers/'
PATH_TO_BENCHMARK_UPS_TUESDAY = PATH_TO_BENCHMARK + 'ups/stop_tuesday.csv'

# File input solver
PATH_TO_SOLVER = PATH_TO_THESIS + 'solver/'
FILE_TO_INPUT_SOLVER_CVRP = PATH_TO_SOLVER + 'cvrp/instances/input_cvrp.vrp'
FILE_TO_INPUT_SOLVER_CVRPTW = PATH_TO_SOLVER + 'cvrptw/instances/input_cvrptw.txt'
FILE_TO_INPUT_SOLVER_UPS = PATH_TO_SOLVER + 'ups/instances/input_cvrptw.txt'

# File output solver
FILE_TO_OUTPUT_SOLVER_CVRP = PATH_TO_SOLVER + 'cvrp/results_routing.txt'
FILE_TO_OUTPUT_SOLVER_CVRPTW = PATH_TO_SOLVER + 'cvrptw/results_routing.txt'
FILE_TO_OUTPUT_SOLVER_UPS = PATH_TO_SOLVER + 'ups/results_routing.txt'

# File storage results
PATH_TO_STORAGE = PATH_TO_THESIS + 'database/'
FILE_TO_STORAGE_CVRP = PATH_TO_STORAGE + 'cvrp/data_resulst_cvrp.csv'
FILE_TO_STORAGE_CVRPTW = PATH_TO_STORAGE + 'cvrptw/data_resulst_cvrptw_new.csv'

# Features files for learning
FILE_TO_TREE_CVRPTW = PATH_TO_STORAGE + 'cvrptw/data_features_cvrptw.csv'

# Storage for Trees
TREE_JSON_CVRPTW = PATH_TO_CONFIGURATION + 'saved_trees/tree_cvrptw.json'
DISPERSION_FEATURES_JSON = PATH_TO_CONFIGURATION + 'saved_trees/dispersion_features.json'

# Storage for stats
STATS_LEARN_OPTIMIZE = PATH_TO_CONFIGURATION + 'saved_stats/stats_learn_optim.csv'

# logs
OUTPUTPATH_LOGS = PATH_TO_CONFIGURATION + "/logs_thesis/"

if not os.path.exists(OUTPUTPATH_LOGS):
    os.mkdir(OUTPUTPATH_LOGS)
    with open(OUTPUTPATH_LOGS + "/.gitignore", "w") as f:
        f.write("*")

FILE_LOG_CLUSTERING = OUTPUTPATH_LOGS + "/clustering_algo.log"
FILE_LOG_COORDINATOR = OUTPUTPATH_LOGS + "/coordinator_algo.log"

