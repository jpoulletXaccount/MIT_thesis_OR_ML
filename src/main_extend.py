
import pandas as pd

from itertools import combinations
from src.helpers import useful_name


def create_all_combination(line):
    """
    From a line of the initial df, derive all potential combinations
    :param line: a line of the intial df
    :return a list of dict which can be easily converted as a df
    """
    data = []
    total_vehi = line[useful_name.NUM_VEHICLE]
    all_vehi = [i for i in range(1,total_vehi +1)]
    for i in all_vehi:
        all_combi = list(combinations(all_vehi,i))

        for combi in all_combi:
            row = dict()
            row[useful_name.INPUT_FILE] = line[useful_name.INPUT_FILE]
            row[useful_name.CLUSTERING_METHOD] = line[useful_name.CLUSTERING_METHOD]
            row[useful_name.TIME_CST] = line[useful_name.TIME_CST]
            row[useful_name.CAPACITY_CST] = line[useful_name.CAPACITY_CST]
            row[useful_name.NUM_VEHICLE] = i
            tot_dist,dist_per_vehi = get_dist_for_combi(line,combi)
            row[useful_name.TOTAL_DISTANCE] = tot_dist
            row[useful_name.STOPS_PER_VEHICLE] = dist_per_vehi
            all_stops_id,stops_per_vehi = get_stop_for_combi(line,combi)
            row[useful_name.STOPS_PER_VEHICLE] = stops_per_vehi
            row[useful_name.ALL_STOPS_ID] = all_stops_id

            data.append(row)

    return data


def get_dist_for_combi(line,combi):
    """
    :param line:
    :param combi:
    :return:
    """
    tot_dist = 0
    dist_per_vehi = ""
    for k in combi:
        tot_dist += float(line[useful_name.DIST_PER_VEHICLE].split("_")[k-1])
        dist_per_vehi += str(line[useful_name.DIST_PER_VEHICLE].split("_")[k-1]) + "_"

    # remove last charactere
    dist_per_vehi = dist_per_vehi[0:len(dist_per_vehi) -1]

    return tot_dist,dist_per_vehi

def get_stop_for_combi(line,combi):
    """
    :param line:
    :param combi:
    :return:
    """
    all_stops_id = ""
    stops_per_vehi = ""
    for k in combi:
        all_stops_id += str(line[useful_name.STOPS_PER_VEHICLE].split("--")[k-1]) + "_"
        stops_per_vehi += str(line[useful_name.STOPS_PER_VEHICLE].split("--")[k-1]) + "--"

    # remove the last chatacters
    all_stops_id = all_stops_id[0:len(all_stops_id)-1]
    stops_per_vehi = stops_per_vehi[0:len(stops_per_vehi)-2]

    return all_stops_id,stops_per_vehi


def extend_dataset(intial_df):
    """
    Recreate all potentials combination
    :param intial_df: the initial dataset
    :return: a new dataset
    """
    all_data = []
    for i,row in intial_df.iterrows():
        all_data.extend(create_all_combination(row))

    extended_results = pd.DataFrame(all_data)
    return extended_results


if __name__ == '__main__':
    list_stops = ['7500']
    for s in list_stops:
        filename = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/" + s + "_customers/interm_results/data_resulst_cvrptw_new_" + s + "_concat.csv"
        df = pd.read_csv(filename)
        output_file = filename.split(".")[0] + '_extended.csv'
        max_nb_vehi= 5
        print("Initial number of instances ", len(df))
        df = df[df[useful_name.NUM_VEHICLE] <=max_nb_vehi]
        print("When reduce to the max number of vehi ", len(df))

        extended_df = extend_dataset(df)
        print("length of the extended dataset ", len(extended_df))
        extended_df.to_csv(output_file,header=True, index=False,mode='w')
