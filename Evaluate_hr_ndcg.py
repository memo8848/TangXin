import math
import heapq
import multiprocessing

import numpy as np
import torch
from joblib._multiprocessing_helpers import mp

from DataLoad import DataLoad


#test_data是 负样本的列表 num_drugs*101维
#test_label是正样本的列表 numdrugs*2维


#eval_hr(test_data)
#对每个drug调用 eval_one_hr(idx,test_data[idx])
#num_drugs个药物hr取平均值


#eval_one_hr(idx,test_target)
#测试一个大小为101的batch 每次输入的batch drug是一样的 target是 那个test——target里面的所有target
#一共101个（一个有反应的 100个没反应的）
#for i in range(num_drugs)
    #drugs= np.full(len(test_target), i, dtype='int64')
    #targets=test_data[idx]
    #pred=model(drugs,targets)
    #pred 排序取前10  求hr 和NDGC

#处理所有药(折)
def evaluate_hr_ndcg(model,w_intMat,test_data,test_label,topK=10):
    global _model
    global _test_data
    global _test_label
    global _topK
    global _w_intMat
    _model = model
    _test_data = test_data
    _test_label = test_label
    _topK = topK
    _w_intMat = w_intMat

    hits, ndcgs = [],[]
    for (idx,label) in test_label:
        #idx是用来评估的用户的下标
        (hr,ndcg) = eval_one_drug(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        #返回一个10维向量 每维度是topk


    return (np.mean(hits), np.mean(ndcgs))





#处理一个药(折) test_data 是100个负样本+1个正样本的列表  test_label是那个正样本
def eval_one_drug(idx):
    # print('药物idx', idx)
    test_data = _test_data[idx]
    test_label =_test_label[idx][1]
    # print(("test_data",test_data))
    # print(("test_label", test_label))
    map_score = {}

    #准备测试数据
    drugs = []
    targets = []
    device = torch.device("cpu")
    for d in np.full(len(test_data), idx, dtype='int64'):
        drugs.append(_w_intMat[d])
    for t in np.array(test_data):
        targets.append(_w_intMat[:, t].T)
    # print("w_intMatd", drugs, "w_intMatt", targets)

    batch_drugs, batch_targets = torch.FloatTensor(np.array(drugs)), torch.FloatTensor(np.array(targets))
    tensor_drugs, tensor_targets = batch_drugs.to(device), batch_targets.to(device)
    y_pred = _model(tensor_drugs, tensor_targets)
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()
    scores = y_pred.flatten()

    #top10
    for i in range(len(test_data)):
        target = test_data[i]
        map_score[target] = scores[i]

    targets.pop()
    # print('靶点：得分', map_score)
    ranklist = heapq.nlargest(_topK, map_score, key=map_score.get)
    # print('前10', ranklist)
    # print('正样本', test_label)
    hr = getHitRatio(ranklist, test_label)
    ndcg = getNDCG(ranklist, test_label)
    return (hr, ndcg)




def getHitRatio(ranklist, test_label):
    for target in ranklist:
        if target == test_label:
            return 1
    return 0

def getNDCG(ranklist, test_label):
    for i in range(len(ranklist)):
        target = ranklist[i]
        if target == test_label:
            return math.log(2) / math.log(i+2)
    return 0




