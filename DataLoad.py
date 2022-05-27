import numpy as np

class DataLoad:
    def __init__(self,data_dir,dataset):
        self.dir = data_dir
        self.dataset = dataset

        self.drug_target_matrix=self.load_y()
        self.num_drugs = self.drug_target_matrix.shape[0]
        self.num_targets=self.drug_target_matrix.shape[1]

        self.drug_similarity=self.load_sd()
        self.target_similarity=self.load_st()


    def load_y(self):
        # return[ [int,int....],[int,int...]....]
        filename = '{}/{}_admat_dgc.txt'.format(self.dir ,self.dataset)
        drug_target_matrix =[]
        with open(filename, "r") as f:
            line = f.readline()
            # 第一行 药物名删掉
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                drug_targets = []
                for x in arr[1:]:
                    # 第一列靶点名删掉
                    drug_targets.append(int(x))
                drug_target_matrix .append(drug_targets)
                line = f.readline()
        return np.array(drug_target_matrix).T

    def load_sd(self):
        # return[ [int,int....],[int,int...]....]
        filename = '{}/{}_simmat_dc.txt'.format(self.dir ,self.dataset)
        # print(filename)
        drug_drug_matrix = []
        with open(filename, "r") as f:
            line = f.readline()
            # 第一行 药物名删掉
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                drug_drugs = []
                for x in arr[1:]:
                    # 第一列靶点名删掉
                    drug_drugs.append(float(x))
                drug_drug_matrix .append(drug_drugs)
                line = f.readline()
        return np.array(drug_drug_matrix)

    def load_st(self):
        # return[ [int,int....],[int,int...]....]
        filename = '{}/{}_simmat_dg.txt'.format(self.dir ,self.dataset)
        # print(filename)
        target_target_matrix = []
        with open(filename, "r") as f:
            line = f.readline()
            # 第一行 药物名删掉
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                target_targets = []
                for x in arr[1:]:
                    # 第一列靶点名删掉
                    target_targets.append(float(x))
                target_target_matrix .append(target_targets)
                line = f.readline()
        return np.array(target_target_matrix)











