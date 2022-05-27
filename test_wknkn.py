from time import time
import sys

import sklearn
from numpy.ma import mean
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from DataLoad import DataLoad as dl
from DrugTargetDataset import DrugTargetDataset as DTD
from DMF import DMF
from collections import defaultdict
from get_negative_samples import get_random_train
from Evaluate_auc_aupr import eval_one_flod
from get_negative_samples import get_reliable_train
from WKNKN import WKNKN
from kNN import knn
from eval_pic import ro_curve
from draw_pic import draw_line

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def cross_validation(intMat,cv,seeds):
    num_flod = 10
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        #num是交叉验证的折数 index是数据总数 step 是一折的数据数
        step = int(index.size/num_flod)
        for i in range(num_flod):
            if i < num_flod-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]

            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)],
                                     dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii],
                                     dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


#分成10折 每一折一个参数 设定k在外围 n在内维算了
#也就是每一折的n  从0.1 到1
# 10个随机种子 每一折10次取平均值
def test_wknkn(data_dir,dataset,cvs):
    t1 = time()
    # 读取原始数据
    data = dl(data_dir, dataset)
    intMat = data.drug_target_matrix
    drugMat = data.drug_similarity
    targetMat = data.target_similarity
    num_drugs, num_targets = intMat.shape[0], intMat.shape[1]
    t2 = time()
    print("header:data, load time:{:.1f}, drugs:{:d}, targets:{:d}"
          .format(t2 - t1, num_drugs, num_targets ))

    # 选择分折方法 拿到训练数据和验证数据
    seeds = [7,90,67,34,15]
    if cvs == 1:
        X, D, T, cv = intMat, drugMat, targetMat, 1
    if cvs == 2:
        X, D, T, cv = intMat, drugMat, targetMat, 0
    if cvs == 3:
        X, D, T, cv = intMat.T, targetMat, drugMat, 0
    cv_data = cross_validation(X, cv, seeds)

    #用训练 得到W intMat*W 用WKNKN 在测试集

    #测试集取全体的 intMat 不行 还是要掩膜一部分 就10折吧 9折用来打分 一折用来测试
    # 分数集合取后来的分数 不用平均 因为都是一样的
    n = 0.5

    _aupr, _auc=[],[]
    for seed in cv_data.keys():  # 5个
        kn=0
        auprs, aucs = [], []
        for W, test_data, test_label in cv_data[seed]:  # 10个 每一折
            #每一折换一个kn
            kn=kn+1
            if cvs == 3:
                W = W.T
            #输入用来打分的是不完整的矩阵9/10
            ww_intMat = WKNKN( W * intMat, drugMat, targetMat, kn, n)
            scores=[]
            #输出用来测试的是不知道的矩阵1/10

            for item in test_data:
                scores.append(ww_intMat[item[0],item[1]])
            print(scores)
            print(test_label)
            prec, rec, thr = precision_recall_curve(test_label,scores)
            # print(" prec", np.array(prec))
            # print(" rec", np.array(prec))
            # print(" thr", np.array(prec))
            sklearn.metrics.precision_recall_curve(test_label, np.array(scores), pos_label=None, sample_weight=None)
            aupr_val = auc(rec, prec)
            fpr, tpr, thr = roc_curve(test_label, np.array(scores))
            auc_val = auc(fpr, tpr)

            aucs.append(auc_val)
            auprs.append(aupr_val)
            print(kn)
        _aupr.append(auprs)
        _auc.append(aucs)
        m_aupr = np.mean(np.array(_aupr), axis=0)
        m_auc = np.mean(np.array(_auc), axis=0)

        print("header:kn:{:.1f},n:{:.1f},aupr:{:.4f},auc:{:4f}".format(kn, n, m_aupr[kn-1], m_auc[kn-1]))



    return

# nn = np.arange(0.1, 1, 0.1)
# for kn in range(10):
#     kn = kn + 1
#     for n in nn:

data_dir="data"
dataset="nr"
test_wknkn(data_dir,dataset,2)
# datasets=["nr","gpcr","ic","e"]
# for dataset in datasets:
#     test_wknkn(data_dir,dataset)







