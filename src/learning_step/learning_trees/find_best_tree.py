import os,copy,sys,json
os.environ["PATH"] += os.pathsep + '/anaconda3/pkgs/graphviz-2.40.1-h69955ae_1/bin/'
sys.path.append(os.getcwd().split("src")[0])

from src.helpers import useful_paths,size_data,useful_name,local_path
from julia import Julia
if "JULIE" in local_path.NAME_MACHINE:
    Julia(runtime="/Applications/Julia-1.3.app/Contents/Resources/julia/bin/julia",compiled_modules=False)
elif "CAMBRIDGE" in local_path.NAME_MACHINE:
    julia_path = os.path.join('C:',os.sep,'Users','jpoullet','AppData','Local','Julia-1.2.0','bin','julia.exe')
    Julia(runtime= julia_path,compiled_modules=False)
else:
    assert "LAGOS" in local_path.NAME_MACHINE, local_path.NAME_MACHINE
    julia_path = os.path.join('C:',os.sep,'Users','jpoullet','AppData','Local','Julia-1.3.1','bin','julia.exe')
    Julia(runtime= julia_path,compiled_modules=False)

from interpretableai import iai

import pandas as pd
import numpy as np


from src.learning_step.learning_trees import constraint
from src.optimization_step.direct_MIP import clustering_cst_parallel_MIP
from src.data_set_creation.stops.manager import stops_manager_cvrptw
from src.learning_step.pixelation import pixelManagerAbsolut


class FindBestTree(object):
    """
    Class which finds the best tree corresponding to the current session and output if as json file
    """
    def __init__(self,file = useful_paths.FILE_TO_TREE_CVRPTW):
        if os.path.exists(file):
            self.df = pd.read_csv(file)
            self.dict_dispersion = self._get_features_dispersion()
        else:
            self.df = None
            self.dict_dispersion = None

        # Stats
        self.score_train = None
        self.score_test = None


    def _get_features_dispersion(self):
        """
        :return: a dict[feature] = variance/mean
        """
        mean_df = self.df.mean(axis=0)
        var_df = self.df.var(axis = 0)

        dict_feat_dispersion = {}
        for f in self.df.columns:
            dict_feat_dispersion[f] = min(1,var_df[f] / max(0.001,abs(mean_df[f])))

        return dict_feat_dispersion


    def set_file_and_df(self,file):
        """
        Change the df accordingly to the given file
        :param file: the file's name
        :return:
        """
        assert os.path.exists(file)
        self.df = pd.read_csv(file)

    def _getX_Y(self):
        """
        Shuflle and separate as X,Y
        :return: a data frame corresponding to X, an other one corresponding to the label class Y
        """
        # shuffle
        self.df = self.df.sample(frac = 1)
        self.dict_dispersion = self._get_features_dispersion()

        # chek the iteration
        if 'iteration' in self.df.columns:
            max_iter = self.df['iteration'].max()
            self.df['iteration'] = self.df['iteration'].apply(lambda x : pow(1,max_iter -x))

        else:
            self.df['iteration'] = [1 for _ in range(0,len(self.df))]

         # separate
        list_col = list(self.df.columns).copy()
        list_col.remove('class_label')


        Y = self.df['class_label']
        X = self.df[list_col]

        assert len(X.columns) == len(list_col)
        assert len(X) == len(Y)
        assert len(X) == len(self.df)

        return X, Y

    def _tune_best_tree(self):
        """
        Tune the tree and then output it as a json file
        :return:
        """
        df_X,df_Y =self._getX_Y()

        # we decompose the tree
        (train_X, train_y), (test_X, test_y) = iai.split_data('classification', df_X, df_Y,seed=1)
        other_colum = list(train_X.columns).copy()
        other_colum.remove('iteration')
        weight_train_X = np.array(train_X['iteration'])
        train_X = train_X[other_colum]
        weight_test_X = np.array(test_X['iteration'])
        test_X = test_X[other_colum]


        grid = iai.GridSearch(iai.OptimalTreeClassifier(random_seed=1),
                              max_depth=[8],
                              # max_depth=[3],
                              minbucket = [10]
                                 )
        # grid.get_learner().set_params(show_progress=False)
        grid.fit(train_X, train_y)

        score_train= grid.score(train_X, train_y, criterion='misclassification')
        score_test = grid.score(test_X, test_y, criterion='misclassification')


        learner = grid.get_learner()
        learner.write_json(useful_paths.TREE_JSON_CVRPTW)
        if 'JULIE' in local_path.NAME_MACHINE:
            learner.write_png(useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/tree_cvrptw.png')

        self.score_test = score_test
        self.score_train = score_train
        print("Accuracy ", score_test, score_train)

        # output the dispersion features corresponding to the df
        f = open(useful_paths.DISPERSION_FEATURES_JSON,'w')
        json.dump(self.dict_dispersion,f)
        f.close()


    @staticmethod
    def get_cst_proba_and_label():
        """
        From the tree previously tune, read it to derive the constraints leading to each leaf
        :return: a tree, a dict[leaf] = list of constraint, a dict[leaf] = label
        """

        learner = iai.read_json(useful_paths.TREE_JSON_CVRPTW)
        if 'JULIE' in local_path.NAME_MACHINE:
            learner.write_png(useful_paths.PATH_TO_CONFIGURATION + 'saved_trees/tree_cvrptw.png')
        num_node = learner.get_num_nodes()
        list_leaf = [i for i in range(1,num_node+1) if learner.is_leaf(i)]
        dict_label = {str(i): learner.get_classification_label(i) for i in list_leaf}
        dict_proba = {str(i) : learner.get_classification_proba(i) for i in list_leaf}

        # build the constraints, which corresponds to each leaf

        dict_constraint = {}
        dict_interm = {'1':[]}

        while len(dict_interm) >= 1:

            new_dict_interm = {}
            for path in dict_interm:
                last_node = int(path.split("-")[-1])
                split_feat = learner.get_split_feature(last_node)
                split_threshold = learner.get_split_threshold(last_node)

                # small child
                child_small = learner.get_lower_child(last_node)
                cst_small = constraint.Constraint(left=split_feat, right=split_threshold, greater=False)
                if learner.is_leaf(child_small):
                    dict_constraint[str(child_small)] = dict_interm[path].copy() + [cst_small]
                else:
                    new_path = path + "-" + str(child_small)
                    new_dict_interm[new_path] = dict_interm[path].copy() + [cst_small]


                # big child
                child_big = learner.get_upper_child(last_node)
                cst_big = constraint.Constraint(left=split_feat, right=split_threshold, greater=True)
                if learner.is_leaf(child_big):
                    dict_constraint[str(child_big)] = dict_interm[path].copy() + [cst_big]
                else:
                    new_path = path + "-" + str(child_big)
                    new_dict_interm[new_path] = dict_interm[path].copy() + [cst_big]

            dict_interm = new_dict_interm

        # for leaf_id in dict_constraint.keys():
        #     list_cst = dict_constraint[leaf_id]
        #     print("Label ",dict_label[leaf_id]," contraint ", [cst.to_print() for cst in list_cst])
        return learner, dict_constraint, dict_label, dict_proba


    def find_cst_best_tree(self):
        """
        Main function of the class: find the the constraints leading to each leaf
        :return: a tree, a dict[leaf] = [list of constraints] and a dict[leaf] = label
        """
        if not os.path.exists(useful_paths.TREE_JSON_CVRPTW):
            self._tune_best_tree()
        elif os.path.exists(useful_paths.DISPERSION_FEATURES_JSON):
            f = open(useful_paths.DISPERSION_FEATURES_JSON,'r')
            self.dict_dispersion = json.load(f)
            f.close()

        if not os.path.exists(useful_paths.DISPERSION_FEATURES_JSON):
            assert False

        tree, dict_cst_computed , dict_leaf_label_computed, dict_proba_computed = self.get_cst_proba_and_label()

        return tree, dict_cst_computed, dict_leaf_label_computed, dict_proba_computed, self.dict_dispersion


    @staticmethod
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



if __name__ == '__main__':
    obj = FindBestTree()
    dict_cst, dict_leaf_label, dict_proba,_ , _= obj.find_cst_best_tree()
    assert False

    # test on one file
    df = pd.read_csv(useful_paths.FILE_TO_STORAGE_CVRPTW)
    for i, row in df.iterrows():
        filename = row[useful_name.INPUT_FILE]
        manager_stop_ref = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename=filename,check_dist_matrix=True)
        list_stop = row[useful_name.ALL_STOPS_ID].split("_")
        manager_stop = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(list_stop,manager_stop_ref)
        manager_pixel = pixelManagerAbsolut.ManagerPixelAbsolut.create_from_manager_stop(manager_stop=manager_stop,dim_pixel=size_data.SIZE_HOMBERGER,nb_pixels=490)
        dist_matrix = obj.get_matrix_pixel(nb_pixel=490)

        # need to duplicate the leaves.
        number_copy = 40
        new_dict_cst = {}
        new_dict_label = {}
        for leaf_id in dict_leaf_label:
            orginal_cst = copy.deepcopy(dict_cst[leaf_id])
            original_label = dict_leaf_label[leaf_id]

            for co in range(0,number_copy):
                new_id = leaf_id +"_co_" + str(co)
                new_dict_cst[new_id] = orginal_cst
                if original_label <=3:
                    new_dict_label[new_id] = original_label
                else:
                    new_dict_label[new_id] = 1000   # we penalized them

        obj_solver = clustering_cst_parallel_MIP.ClusteringCstParallelMIP(managerPixel=manager_pixel,
                                                                          manager_stop=manager_stop,
                                                                          matrix_dist_pix=dist_matrix,
                                                                          dict_leaf_cst=new_dict_cst,
                                                                          dict_leaf_label= new_dict_label)
        dict_leaf_stop = obj_solver.create_relevant_clusters()
        min_number = sum(new_dict_label[leaf_id] for leaf_id in dict_leaf_stop)

        assert min_number >= row[useful_name.NUM_VEHICLE], print(min_number,row[useful_name.NUM_VEHICLE],i)




