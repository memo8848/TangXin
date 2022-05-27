import re
from DMF import DMF
import os
from DataLoad import DataLoad as dl


def load_model_name():
    model_dir='pretrain'
    file_names = os.listdir(model_dir)
    file_1 = file_names[0]
    file_1_split = file_1.split('-')
    print('file1', file_1_split)
    dataset=file_1_split[1]
    cvs = file_1_split[2]
    layers= re.findall('\[(.*)\]',file_1_split[4])
    print('layers', layers[0])
    layers=list(map(int, layers[0].split(",")))
    return file_1,dataset,cvs,layers

#加载模型
def load_model():
    save_dict = 'D:/evaluate/auc'
    #加载原始数据
    file_name,dataset,cvs,layers = load_model_name()
    data_dir='data'
    data = dl(data_dir, dataset)
    intMat = data.drug_target_matrix
    num_drugs, num_targets = intMat.shape[0], intMat.shape[1]

    #加载训练好的模型
    model = DMF(num_drugs, num_targets, layers)
    model.load_state_dict(torch.load('pretrain/'+file_name))
    print('loading pretrain model....'+file_name)
    return model