

class ManagerCluster(dict):

    def __init__(self):
        super(ManagerCluster,self).__init__()


    def add_cluster(self,clus):
        assert not clus.guid in self.keys(),clus.guid
        self[clus.guid] = clus




