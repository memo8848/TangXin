import numpy as np
from kNN import knn
from DataLoad import DataLoad as dl
from k_nearest_neighbor import k_nearest
#生成训练数据
#随机选择负样本，返回训练数据，一个正样本，num_negative个负样本，格式[ [drug,drug....],[target,target...],[label,label...]]
def get_random_train(num_negative,w_intMat):
    drugs, targets, labels = [], [], []
    train_data = np.argwhere(w_intMat == 1)
    train_dict = {}

    for i in train_data:
        train_dict[(i[0], i[1])] = 1
    for i in train_data:
        drugs.append(i[0])
        targets.append(i[1])
        labels.append(1)
        for t in range(num_negative):
            j = np.random.randint(w_intMat.shape[1])
            while (i[0], j) in train_dict:
                j = np.random.randint(w_intMat.shape[1])
            drugs.append(i[0])
            targets.append(j)
            labels.append(0)
    return [np.array(drugs), np.array(targets), np.array(labels)]


# 因为有些drug可能一个可靠负样本都没有 最低取不取负样本 最高取num_negative 个
def get_reliable_train(k,w_intMat,drugMat,targetMat):
    drug, target, labels = [], [], []
    train_data = np.argwhere(w_intMat == 1)
    train_dict = {}

    #train_dict 的键 是（药物，靶点） 值是1
    for i in train_data:
        train_dict[(i[0], i[1])] = 1

    # negatives_all = knn(k,num_negative,w_intMat,drugMat,targetMat)


    #添加有反应的药物序号,正样本们 靶点序号 相互作用序号 drug=[drug],target=[target],labels=[1]
    # for i in train_data:
    #     drug.append(i[0])
    #     target.append(i[1])
    #     labels.append(1)
    #     for item in negatives_all[i]:
    #         drug.append(i[0])
    #         target.append(item[1])
    #         labels.append(0)

    #添加可靠负样本们 一口气添加到正样本后面
    negatives_all = k_nearest(k, w_intMat, drugMat, targetMat)
    for i in train_data:
        drug.append(i[0])
        target.append(i[1])
        labels.append(1)
    for item in negatives_all:
        drug.append(item[0])
        target.append(item[1])
        labels.append(0)


    # print([np.array(drug), np.array(target), np.array(labels)])
    return [np.array(drug), np.array(target), np.array(labels)]



# data=dl('data','nr')
# get_reliable_train(7,data.drug_target_matrix,data.drug_similarity,data.target_similarity)








