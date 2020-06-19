from src.helpers import useful_paths

import logging
from logging import FileHandler
from logging import Formatter

LOG_LEVEL = "INFO"
LOG_FORMAT = ("%(asctime)s [%(levelname)s]: %(message)s")

# ====================== CLUSTERING ALGO LOGGER ======================
# Log with the clustering algo specific information
CLUSTERING_LOG_FILE = useful_paths.FILE_LOG_CLUSTERING
clustering_logger = logging.getLogger("thesis.clusteringlog")

clustering_logger.setLevel(LOG_LEVEL)
clustering_file_handler = FileHandler(CLUSTERING_LOG_FILE)
clustering_file_handler.setLevel(LOG_LEVEL)
clustering_file_handler.setFormatter(Formatter(LOG_FORMAT))
clustering_logger.addHandler(clustering_file_handler)
clustering_logger.propagate = False


# ====================== COORDINATOR ALGO LOGGER ======================
# Log with the clustering algo specific information
COORDINATOR_LOG_FILE = useful_paths.FILE_LOG_COORDINATOR
coordinator_logger = logging.getLogger("thesis.coordinatorlog")

coordinator_logger.setLevel(LOG_LEVEL)
coordinator_file_handler = FileHandler(COORDINATOR_LOG_FILE)
coordinator_file_handler.setLevel(LOG_LEVEL)
coordinator_file_handler.setFormatter(Formatter(LOG_FORMAT))
coordinator_logger.addHandler(coordinator_file_handler)
coordinator_logger.propagate = False



def reset_all():
    list_log = [coordinator_logger,clustering_logger]
    for log in list_log:
        reset_log(log.handlers[0].baseFilename)
        log.info("\n\n")
        log.info("==========================================================================================")
        log.info("==========================================================================================")
        log.info("==========================================================================================")
        log.info("================================     NEW RUN START    ====================================")
        log.info("==========================================================================================")
        log.info("==========================================================================================")

def reset_log(filename):
     with open(filename, "w"):
        pass
