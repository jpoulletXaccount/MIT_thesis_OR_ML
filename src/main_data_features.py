import os, sys
sys.path.append(os.getcwd().split("src")[0])


from src.helpers import useful_paths,useful_name,customs_exception,size_data
from src.data_set_creation.stops.manager import stops_manager_cvrp, stops_manager_cvrptw,stops_manager_cvrptw_ups
from src.learning_step.pixelation import pixelManagerAbsolut,pixelManagerRelative
from src.learning_step.learning_trees.features_creation import derivingAllFeatures, derivingFeaturesCreatedDataset, \
    derivingFeaturesUPS, derivingFeaturesPixelRelatives, derivingFeaturesPixel, derivingFeaturesCreatedDatasetLinear

import pandas as pd
from tqdm import tqdm


def retrieve_corresponding_stops_manager(row,reference_manager):
    """
    Retrive an object stops_manager
    :param row: the corresponding row
    :return: a manager stop object
    """
    list_of_list = row[useful_name.STOPS_PER_VEHICLE].split('--')
    list_of_stops = []
    for l in list_of_list:
        list_of_stops.extend(l.split('_'))

    if 'cvrptw' in row[useful_name.INPUT_FILE]:
        return stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(list_of_stops, reference_manager)
    elif 'cvrp' in row[useful_name.INPUT_FILE]:
        return stops_manager_cvrp.StopsManagerCRVP.from_sublist(list_of_stops, reference_manager)
    elif 'ups' in row[useful_name.INPUT_FILE]:
        return stops_manager_cvrptw_ups.StopsManagerCVRPTW_UPS.from_sublist(list_of_stops,reference_manager)
    else:
        raise customs_exception.WrongFile("we currently only deal with cvrp and cvrptw and ups")



def get_reference_manager(filename):
    """
    Retrieve the reference manager (i.e.) corresponding to all stops for the filename
    :param filename: the corresponding filenames
    :return: a stops manager
    """
    if 'cvrptw' in filename:
        ref_manager = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename,check_dist_matrix=True)
    elif 'cvrp' in filename:
        ref_manager = stops_manager_cvrp.StopsManagerCRVP.from_cvrp_file(filename,check_dist_matrix=True)
    elif 'ups' in filename:
        ref_manager = stops_manager_cvrptw_ups.StopsManagerCVRPTW_UPS.from_ups_file(filename,date=None)
    else:
        raise customs_exception.WrongFile

    return ref_manager

def get_matrix_pixel(nb_pixel):
    """
    :param nb_pixel: the nb pixel we are aiming for
    :return: a dict[pixelid][pixelid]
    """
    empty_manager_pixel = pixelManagerAbsolut.ManagerPixelAbsolut.create_empty_pixel(size_data.SIZE_HOMBERGER, nb_pixel,depot=None)
    list_pixel_id = list(empty_manager_pixel.keys())
    dict_matrix = {pixelId : {} for pixelId in list_pixel_id}

    for i in range(0,len(list_pixel_id)):
        pixel_id_1 = list_pixel_id[i]
        pixel_1 = empty_manager_pixel[pixel_id_1]

        for j in range(i, len(list_pixel_id)):
            pixel_id_2 = list_pixel_id[j]
            pixel_2 = empty_manager_pixel[pixel_id_2]

            dist = pixel_1.get_distance_to_point(pixel_2)
            dict_matrix[pixel_id_1][pixel_id_2] = dist
            dict_matrix[pixel_id_2][pixel_id_1] = dist


    return dict_matrix

def derive_features_pixel_relative(manager_stops,nb_pixel):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param manager_stops: a manager stop containing all stops
    :param nb_pixel: the number of pixels we are interested in
    :return: a dict['feature name'] = feature value
    """
    manager_pixel = pixelManagerRelative.ManagerPixelRelative.create_from_manager_stop(manager_stops, length=5, nb_pixels=nb_pixel)

    features_object = derivingFeaturesPixel.DerivingFeaturesPixel(manager_pixel)

    return features_object.derive_features()

def derive_features_pixel_absolut(manager_stops,nb_pixel):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param nb_pixel: the number of pixels we are interested in
    :param manager_stops: a manager stop containing all stops
    :return: a dict['feature name'] = feature value
    """
    manager_pixel = pixelManagerAbsolut.ManagerPixelAbsolut.create_from_manager_stop(manager_stops, size_data.SIZE_HOMBERGER, nb_pixel)

    features_object = derivingFeaturesPixel.DerivingFeaturesPixel(manager_pixel)

    return features_object.derive_features()

def derive_features_relatives_pixel_absolut(manager_stops,nb_pixel,matrix_pix):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param manager_stops: a manager stop containing all stops
    :return: a dict['feature name'] = feature value
    """
    manager_pixel = pixelManagerAbsolut.ManagerPixelAbsolut.create_from_manager_stop(manager_stops, size_data.SIZE_HOMBERGER, nb_pixel)
    features_object = derivingFeaturesPixelRelatives.DerivingFeaturesPixelRelatives(manager_pixel, matrix_pix)

    return features_object.derive_features()


def derive_feature_best(manager_stops):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param manager_stops: a manager stop containing all stops
    :return: a dict['feature name'] = feature value
    """
    feature_object = derivingAllFeatures.DerivingBestFeatures(manager_stops)

    return feature_object.derive_features()

def derive_feature_created(manager_stops):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param manager_stops: a manager stop containing all stops
    :return: a dict['feature name'] = feature value
    """
    feature_object = derivingFeaturesCreatedDataset.DerivingFeaturesCreatedDataset(manager_stops)

    return feature_object.derive_features()

def derive_feature_ups(manager_stops):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param manager_stops: a manager stop containing all stops
    :return: a dict['feature name'] = feature value
    """
    feature_object = derivingFeaturesUPS.DerivingFeaturesUPS(manager_stops)

    return feature_object.derive_features()


def derive_feature_created_linear(manager_stops):
    """
    From the manager containing all the relevant stops, derive the relevant aggregated features
    :param manager_stops: a manager stop containing all stops
    :return: a dict['feature name'] = feature value
    """
    feature_object = derivingFeaturesCreatedDatasetLinear.DerivingFeaturesCreatedDatasetLinear(manager_stops)

    return feature_object.derive_features()

def build_data_set_features_for_filename(data_frame,output_file):
    """
    Froma a data frame object, read all the rows and convert them into a meaningful data set
    :param data_frame: a df pandas object, subset of previous experiment
    :param output_file: the resulting file, ready to be trained
    """
    list_filename = list(data_frame[useful_name.INPUT_FILE].unique())
    assert len(list_filename) == 1, list_filename
    ref_manager = get_reference_manager(list_filename[0])
    nb_pixel = 490
    #nb_pixel = 100
    matrix_pixel = get_matrix_pixel(nb_pixel)

    data = []
    for i,row in data_frame.iterrows():
        stop_manager = retrieve_corresponding_stops_manager(row,ref_manager)
        # dict_features = derive_feature_best(stop_manager)
        dict_features = derive_feature_created(stop_manager)
        # dict_features = derive_feature_ups(stop_manager)
        # dict_features = derive_feature_created_linear(stop_manager)
        #dict_features = derive_features_relatives_pixel_absolut(stop_manager,nb_pixel,matrix_pixel)
        #dict_features = derive_features_pixel_relative(stop_manager,nb_pixel)
        #dict_features = derive_features_pixel_absolut(stop_manager,nb_pixel)
        dict_features['class_label'] = correct_label(row[useful_name.NUM_VEHICLE])
        data.append(dict_features)

    header = True
    if os.path.exists(output_file):
        header = False

    data_results = pd.DataFrame(data)
    data_results.to_csv(output_file, header=header, index=False,mode = "a")

def correct_label(label):
    """
    Correct the label is not accurate enough (we typically do not want more than 3 vehicles.
    So everything above is going to be a 4
    :param label: the current label
    :return: the "true" label
    """
    if int(label) >= 6:
        return 6
    else:
        return label

def build_data_set_features(data_frame,output_file):
    """
    Froma a data frame object, read all the rows and convert them into a meaningful data set
    :param data_frame: a df pandas object, subset of previous experiment
    :param output_file: the resulting file, ready to be trained
    """
    list_filename = list(data_frame[useful_name.INPUT_FILE].unique())
    for filename in tqdm(list_filename):
        df_file = data_frame[data_frame[useful_name.INPUT_FILE] == filename]
        build_data_set_features_for_filename(df_file,output_file=output_file)

def get_diameter_filename(df_file):
    """
    :param df_file: the df corresponding to one specific file
    :return: the list of diameters corresponding to the different subset of stops
    """
    list_dia = []
    list_filename = list(df_file[useful_name.INPUT_FILE].unique())
    assert len(list_filename) == 1, list_filename
    ref_manager = get_reference_manager(list_filename[0])
    for i,row in df_file.iterrows():
        stop_manager = retrieve_corresponding_stops_manager(row,ref_manager)
        list_dia.append(stop_manager.get_diameter())

    return list_dia


def get_list_diameters(data_frame):
    """
    :param data_frame: a data frame corresponding to
    :return: the list of diameters corresponding to the different subset of stops
    """
    list_dia = []
    list_filename = list(data_frame[useful_name.INPUT_FILE].unique())
    for filename in tqdm(list_filename):
        df_file = data_frame[data_frame[useful_name.INPUT_FILE] == filename]
        list_dia += get_diameter_filename(df_file)

    return list_dia


if __name__ == '__main__':
    df = pd.read_csv("/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/10000_customers/interm_results/data_resulst_cvrptw_new_10000_concat_extended.csv")
    #df = pd.read_csv("/Users/jpoullet/Documents/MIT/Thesis/database/ups/data_resulst_cvrptw_new.csv")
    print(len(df))
    size_data.NUMBER_CUSTOMERS = 600

    # df = df[df[useful_name.CLUSTERING_METHOD] == 'ClusteringMethod.KMEAN']
    #df = df[df[useful_name.INPUT_FILE].str.contains("R1_")]
    # print(len(df))
    #df = df[df[useful_name.CLUSTERING_METHOD] == 'ClusteringMethod.RANDOM']

    if os.path.exists(useful_paths.FILE_TO_TREE_CVRPTW):
        os.remove(useful_paths.FILE_TO_TREE_CVRPTW)

    print(df[useful_name.CAPACITY_CST].unique())
    print(df[useful_name.TIME_CST].unique())
    for cap in list(df[useful_name.CAPACITY_CST].unique()):
        # if cap != 200:
        #     continue
        df_cap = df[df[useful_name.CAPACITY_CST] == cap]

        # df_cap = df_cap[df_cap[useful_name.TIME_CST] == 1206]
        assert len(df_cap[useful_name.TIME_CST].unique()) == 1, df_cap[useful_name.TIME_CST].unique()
        time_cst = list(df_cap[useful_name.TIME_CST].unique())[0]
        print(cap,time_cst)

        build_data_set_features(data_frame=df_cap,
                                output_file=useful_paths.FILE_TO_TREE_CVRPTW)
    #     list_diameter = get_list_diameters(df_cap)
    #
    # print("max ", max(list_diameter))
    # print("min ", min(list_diameter))
    # print("60 percentile", np.percentile(list_diameter,60))
    # print("75 percentile", np.percentile(list_diameter,75))
    # print("90 percentile", np.percentile(list_diameter,90))
    # print("mean ", np.mean(list_diameter))

