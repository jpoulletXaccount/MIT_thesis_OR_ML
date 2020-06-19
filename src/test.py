
import pandas as pd

from src.data_set_creation.stops.manager import manager_packages,stops_manager_cvrptw_ups_project
from src.learning_step.learning_trees.features_creation import derivingFeaturesUPS

from src.helpers import useful_paths

if __name__ == '__main__':
    df = pd.read_csv("/Users/jpoullet/Desktop/UPS/UPS2018NEW/results_routing_ml_concat.csv")

    file_pack = "/Users/jpoullet/Dropbox (MIT)/UPS Project/09 Final Project/01 - UPS inputs/01 Base Data/DR17_0120_Scenarios/data_for_packages.csv"
    ref_packages = manager_packages.ManagerPackages.from_package_ups_file(date=df['Date'].unique(),filename=file_pack)
    print("Number of packages ", len(ref_packages))


    file_stop = "/Users/jpoullet/Dropbox (MIT)/UPS Project/09 Final Project/01 - UPS inputs/01 Base Data/DR17_0120_Scenarios/deliveries_and_pickups.csv"
    ref_stops = stops_manager_cvrptw_ups_project.StopsManagerCVRPTW_UPS_Project.from_ups_file_with_pack(filename=file_stop,date=df['Date'].unique(),manager_package=ref_packages)
    print("Number of stops ", len(ref_stops))

    data = []
    for i,row in df.iterrows():
        row_pack_stop = row['sequence_tuple_stop_init_list_package_id']
        specific_manager = stops_manager_cvrptw_ups_project.StopsManagerCVRPTW_UPS_Project.from_sublist(row_stops_pack=row_pack_stop,reference_manager_stop=ref_stops)
        featurer = derivingFeaturesUPS.DerivingFeaturesUPS(specific_manager, project=True)
        dict_feature = featurer.derive_features()
        dict_feature['class_label'] = row['nb_vehicles']
        data.append(dict_feature)

    data_results = pd.DataFrame(data)
    data_results.to_csv(useful_paths.FILE_TO_TREE_CVRPTW, index=False,mode = "w")





