
from src.data_set_creation.stops.manager import stops_manager_cvrp,stops_manager_cvrptw,stops_manager_cvrptw_ups
from src.data_set_creation.config import config_constraint_cvrp
from src.data_set_creation.config import config_constraint_cvrptw
from src.data_set_creation.routing_model import cvrptw_routing_solver, cvrp_routing_solver,ups_routing_solver
from src.data_set_creation.clustering_process import perform_clustering
from src.helpers import useful_paths,customs_exception,local_path

from tqdm import tqdm

import os,smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd


def send_email(msg):
    """
    Send an email when the program has either crashed or finished
    :param msg: the message content
    :return:
    """
    # Create a secure SSL context
    context = ssl.create_default_context()

    # set up the SMTP server
    s = smtplib.SMTP_SSL(host="smtp.gmail.com", port=465,context=context)
    s.login(local_path.EMAIL_SENDER_ADDRESS, local_path.EMAIL_PASSWORD)

    email = MIMEMultipart()
    email['From'] = local_path.EMAIL_SENDER_ADDRESS
    email['To'] = local_path.EMAIL_RECEIVER_ADDRESS
    email['Subject'] = msg
    email.attach(MIMEText(msg,'plain'))

    s.send_message(email)

def read_stop(filename,date):
    """
    Read the filename
    :param filename: corresponding path
    :param date: corresponding date, useful for UPS
    :return: a manager stop object
    """
    if 'cvrptw' in filename:
        return read_stops_cvrptw(filename)
    elif 'cvrp' in filename:
        return read_stops_cvrp(filename)
    elif 'ups' in filename:
        return read_stops_ups(filename,date)

def read_stops_ups(filename,date):
    """
    Create a stop_manager_cvrptw_ups objcet
    :param filename: the filename
    :param date: the corresponding date
    :return: such an object
    """
    return stops_manager_cvrptw_ups.StopsManagerCVRPTW_UPS.from_ups_file(filename,date)

def read_stops_cvrp(filename):
    """
    Create a stop_manager_cvrp objcet
    :param filename: the filename
    :return: such an object
    """
    return stops_manager_cvrp.StopsManagerCRVP.from_cvrp_file(filename,check_dist_matrix=True)

def read_stops_cvrptw(filename):
    """
    Create a stop_manager_cvrptw objcet
    :param filename: the filename
    :return: such an object
    """
    return stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)

def read_config(filename):
    """
    Create a config constraint object
    :param filename: the filename corresponding
    :return: such an object
    """
    if 'cvrptw' in filename:
        return read_config_cvrptw(filename)
    elif 'cvrp' in filename:
        return read_config_cvrp(filename)
    elif 'ups' in filename:
        return read_config_ups(filename)

def read_config_cvrp(filename):
    """
    Creare a config_cst_cvrp object
    :param filename: the corresponding filename
    :return: such an object
    """
    return config_constraint_cvrp.ConfigCVRP.from_cvrp_file(filename)

def read_config_ups(filename):
    """
    Creare a config_cst_cvrptw object
    :param filename: the corresponding filename
    :return: such an object
    """
    config_object = config_constraint_cvrptw.ConfigCVRPTW(cap=350,filename=filename,nb_vehi=100)
    return config_object

def read_config_cvrptw(filename):
    """
    Creare a config_cst_cvrptw object
    :param filename: the corresponding filename
    :return: such an object
    """
    return config_constraint_cvrptw.ConfigCVRPTW.from_cvrptw_file(filename)

def get_solver_object(filename,manager_stop,config_cst):
    """
    Choose the right solver for the type of problem
    :param filename: the filename
    :param manager_stop: corresponding manager_stop
    :param config_cst: corresponding congig_cst
    :return:
    """
    if 'cvrptw' in filename:
        return cvrptw_routing_solver.RoutingSolverCVRPTW(manager_stop, config_cst)
    elif 'cvrp' in filename:
        return cvrp_routing_solver.RoutingSolverCVRP(manager_stop, config_cst)
    elif 'ups' in filename:
        return ups_routing_solver.RoutingSolverUPS(manager_stop, config_cst)
    else:
        raise customs_exception.FunctionalityNotImplemented


def solve_problem(filename,clustering_name,date=None):
    """
    Solve the routing problem
    :param filename: the file from which we want to solve routing problem
    :param clustering_name: the clustering method to be used
    :return:
    """

    manager_stops = read_stop(filename,date)
    assert manager_stops.check_demand_updated()
    config_object = read_config(filename)
    config_object.cluster_method = clustering_name
    clustering_object = perform_clustering.StopClustering()
    list_cluster = clustering_object.perform_clustering(manager_stops,config_object)

    for cluster in list_cluster:
        # if tw:
        #     current_manager_stops = stops_manager_cvrptw.StopsManagerCVRPTW.init_from_cluster(cluster)
        # elif ups:
        #     current_manager_stops = stops_manager_cvrptw_ups.StopsManagerCVRPTW_UPS.init_from_cluster(cluster)
        # else:
        #     current_manager_stops = stops_manager_cvrp.StopsManagerCRVP.init_from_cluster(cluster)
        current_manager_stops = manager_stops.init_from_cluster(cluster)
        current_manager_stops.set_depot(manager_stops.depot)
        solver_obj = get_solver_object(filename,current_manager_stops,config_object)
        solver_obj.solve_routing()


if __name__ == '__main__':
    list_filename = []
    nb_customers = '1000_customers/'
    path_dir = useful_paths.PATH_TO_BENCHMARK_CREATED + nb_customers
    for r, d, f in os.walk(path_dir):
        for file in f:
            if '.txt' in file:
                list_filename.append(os.path.join(r, file))
    print(len(list_filename), " benchmarks instances to be trained on")

    date = None
    comp = 0
    for filename in tqdm(list_filename):
        comp +=1
        # df = pd.read_csv(filename)
        # for date in tqdm(list(df['Date'].unique())):
        clustering_name = "KMEAN"
        solve_problem(filename,clustering_name,date)
        clustering_name = "RANDOM"
        solve_problem(filename,clustering_name,date)

    if local_path.SEND_EMAIL:
        msg = 'Finished creating data for ' + nb_customers
        send_email(msg)

