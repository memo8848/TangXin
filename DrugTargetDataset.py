import torch
from torch.utils.data import Dataset

class DrugTargetDataset(Dataset):
    def __init__(self, drug_indics, target_indics,label,drug_target_matrix):
        self.drug_target_matrix = torch.FloatTensor(drug_target_matrix)
        self.drug = torch.LongTensor(drug_indics)
        self.target = torch.LongTensor(target_indics)
        self.interaction = torch.FloatTensor(label)

    def __getitem__(self, index):
        drug = self.drug_target_matrix[self.drug[index]]
        target = self.drug_target_matrix[:,self.target[index]].t()
        interaction=self.interaction[index]
        return drug, target, interaction

    def __len__(self):
        return self.drug.size(0)