

class Constraint(object):
    """
    Corresponds to a linear constraint from the tree
    """
    def __init__(self,left,right, greater):
        self.left_side = left
        self.right_side = right
        self.is_greater = greater       # If True then it means that the left_side is greater than the right_size


    def to_print(self):
        """
        :return: a string ready to be printed
        """
        text = str(self.left_side)
        if self.is_greater:
            text += " >= "
        else:
            text += " <= "

        text += str(self.right_side)
        return text


    def is_respected(self,dict_feature, threshold = 0):
        """
        :param threshold: the level of robustness we want to tackle
        :param dict_feature:
        :return: a boolean indicating is the constraint is satisfied or not
        """
        if self.left_side.startswith('x'):
            attr_nu = int(self.left_side[1:])
            val_left = dict_feature[list(dict_feature.keys())[attr_nu-1]]
        else:
            val_left= dict_feature[self.left_side]

        if self.is_greater:
            return val_left >= (1 + threshold) *self.right_side
        else:
            return val_left < (1 - threshold) * self.right_side


    def get_feature_name(self,dict_feature):
        """
        return the name of the features corresponding to the constraints
        :param dict_feature: a dict feature
        :return:
        """
        if self.left_side.startswith('x'):
            attr_nu = int(self.left_side[1:])
            attr_name = list(dict_feature.keys())[attr_nu-1]
            assert False
        else:
            attr_name = self.left_side

        return attr_name


    def get_margin(self,dict_feature,threshold=0):
        """
        :param dict_feature: a dict of features
        :param threshold: the level of robustness we want to add
        :return: a dict[feat_name] = margin
        """
        feat_name = self.get_feature_name(dict_feature)
        if self.is_greater:
            margin = self.right_side * (1 + threshold) - dict_feature[feat_name]
            margin = min(0,margin)
        else:
            margin = self.right_side * (1 - threshold) - dict_feature[feat_name]
            margin = max(0,margin)

        assert margin <=0 if self.is_greater else margin >=0, margin
        return {feat_name:margin}


