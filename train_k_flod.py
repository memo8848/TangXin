from time import time
import sys

from numpy.ma import mean
import numpy as np

from DataLoad import DataLoad as dl
from DrugTargetDataset import DrugTargetDataset as DTD
from DMF import DMF
from BDMF import BDMF
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


#获得交叉验证的训练和测试数据  train=W*intMat得到训练用矩阵(其中把test对应的intMat[d][t]设置为0)
#cv=1 随机抽drug-target对  cv=0 随机抽drug行或target列
#注意cvs=3时，输出的W test_data中drug和target位置相反，需要调换
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

#训练 一个epoch 训练一个模型 对这个模型进行5折交叉验证 每折apur平均 每次交叉验证也平均aupr  得到该epoch的aupr
def train_k_flod(para,data_dir,dataset,cvs,num_negative,k=0,kn=0,n=0):
    t1=time()
    # 读取原始数据
    data=dl(data_dir,dataset)
    intMat=data.drug_target_matrix
    drugMat=data.drug_similarity
    targetMat=data.target_similarity
    num_drugs, num_targets = intMat.shape[0], intMat.shape[1]
    t2=time()
    print("header:data, load time:{:.1f}, drugs:{:d}, targets:{:d}"
          .format(t2 - t1, num_drugs,  num_targets,))

    #选择分折方法 拿到训练数据和验证数据
    seeds = [7, 83, 22, 18, 41]
    if cvs == 1:
        X, D, T, cv = intMat, drugMat, targetMat, 1
    if cvs == 2:
        X, D, T, cv = intMat, drugMat, targetMat, 0
    if cvs == 3:
        X, D, T, cv = intMat.T, targetMat, drugMat, 0
    cv_data = cross_validation(X, cv,seeds)

    #构建模型
    # model = DMF(num_drugs, num_targets, para.layers)
    model = BDMF(num_drugs, num_targets, para.layers)
    model.weight_init()

    #log损失函数
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.reg)
    model_name = str(model.__class__)[17:][:-2]
    print(model_name)
    loss_name = str(criterion.__class__)[17 + 13:][:-2]
    print(
        'header:class:{},dataset:{}, batch_size:{}, epochs:{}, latent_dim:{}, num_negative:{}, lr:{}, reg:{},loss:{}'
            .format(model_name, dataset, para.batch_size, para.epochs, para.latent_dim, num_negative, para.lr, para.reg,
                    loss_name))


    # 不训练，进行5次10折交叉验证，空白对照
    t1 = time()
    auprs, aucs = [], []
    for seed in cv_data.keys():
        _auprs = []
        _aucs = []
        for W, test_data, test_label in cv_data[seed]:
            if cvs==3:
                W=W.T
            (_,_,aupr_val, auc_val) = eval_one_flod(model,cvs,test_data, test_label,W * intMat)
            _auprs.append(aupr_val)
            _aucs.append(auc_val)
        auprs.append(mean(auprs))
        aucs.append(mean(aucs))
    aupr = mean(auprs)
    auc = mean(aucs)
    t2 = time()
    print('epoch:0,train_time:{:.1f}s, aupr:{:.4f}, auc:{:.4f}, test_time:{:.1f}s'.format(t1 - t1, aupr, auc, t2 - t1))
    model_out_file = 'pretrain/' + '/{}-{}-{}-{}-{}-{}-lr_{}-AUPR_{:.4f}-AUC_{:.4f}-model_{}.model'.format(para.method,
                                                                                                       dataset,
                                                                                                       cvs,
                                                                                                       para.latent_dim,
                                                                                                       para.layers,
                                                                                                       num_negative,
                                                                                                       para.lr,
                                                                                                       aupr,
                                                                                                       auc,
                                                                                                       0)
    if para.out > 0:
        torch.save(model.state_dict(), model_out_file)

    best_aupr, best_auc, best_iter, best_epoch = 0, 0, -1, -1
    count = 0
    auc_list=[]
    for epoch in range(para.epochs):
        epoch=epoch+1
        auprs, aucs = [],[]
        fprs, tprs = [], []

        #fprs分成100份
        mean_fpr = np.linspace(0, 1, 50)
        t1 = time()
        for seed in cv_data.keys():#5个
            _auprs, _aucs = [], []
            _fprs,_tprs=[],[]
            for W, test_data, test_label in cv_data[seed]:#10个 每一折
                if cvs == 3:
                    W = W.T
                model.train()
                # 填装训练数据
                #用WKNN
                if kn == 0:
                    train = get_random_train(num_negative, W * intMat)
                else:
                #用KNN
                    ww_intMat = WKNKN(W * intMat, drugMat, targetMat, kn, n)
                    train = get_random_train(num_negative, ww_intMat)
                # print(train)
                train = DTD(train[0], train[1], train[2], W * intMat)
                train = DataLoader(train, batch_size=para.batch_size, shuffle=True)
                t1 = time()
                #训练9折
                for batch_idx, (drug, target, y) in enumerate(train):
                    y_hat = model(drug, target)
                    loss = criterion(y_hat, y.view(-1, 1),)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t2 = time()

                #测试一折
                model.eval()
                if kn==0:
                    (fpr_val,tpr_val,aupr_val,auc_val) = eval_one_flod(model,cvs,test_data, test_label, W * intMat)
                else:
                    ww_intMat =WKNKN(W * intMat,drugMat,targetMat,kn,n)
                    (fpr_val,tpr_val,aupr_val, auc_val) = eval_one_flod(model, cvs, test_data, test_label, ww_intMat)
                #10个折的列表

                _auprs.append(aupr_val)
                _aucs.append(auc_val)

                # 每折里的个数不统一 用roc切统一了
                mean_tpr = np.interp(mean_fpr, fpr_val, tpr_val)
                _fprs.append(mean_fpr)
                _tprs.append(mean_tpr)


            #1次10折的
            auprs.append(mean(_auprs))
            aucs.append(mean(_aucs))
            fprs.append(np.mean(np.array(_fprs),axis=0))
            tprs.append(np.mean(np.array(_tprs),axis=0))

        #5次10折的平均
        aupr=mean(auprs)
        auc=mean(aucs)
        #肯定是整齐的
        fpr=np.mean(np.array(fprs),axis=0)
        tpr=np.mean(np.array(tprs),axis=0)

        auc_list.append(auc)
        #把损失存存 转化为float保留俩位小数 画图
        if epoch % para.verbose == 0:
            print('epoch:{},train_time:{:.1f}s, aupr:{:.4f}, auc:{:.4f}, test_time:{:.1f}s, loss:{:.6f}'
                              .format(epoch, t2 - t1, aupr, auc, time() - t2, loss))
        if auc > best_auc :
            count = 0
            best_train_time, best_aupr, best_auc, best_epoch, best_test_time = t2 - t1,aupr, auc, epoch, time() - t2
            file_name='/{}-{}-{}-{}-{}-{}-lr_{}-AUPR_{:.4f}-AUC_{:.4f}-epoch_{}.model'.format(
                                    para.method,
                                    dataset,
                                    cvs,
                                    para.latent_dim,
                                    para.layers,
                                    num_negative,
                                    para.lr,
                                    aupr,
                                    auc,
                                    epoch)
            model_out_file = 'pretrain/' + file_name

            #画roc
            # ro_curve(fpr,tpr, "AUC/" + file_name, para.method,dataset)



            if para.out > 0:
                torch.save(model.state_dict(), model_out_file)
        else:
            count += 1
        if count >= 50:
            print('best epoch:{},aupr:{:.4f}, auc:{:.4f}'.format(best_epoch, best_aupr, best_auc))
            if para.out > 0:
                print("The best model is saved to {}".format(model_out_file))
            return auc_list
            sys.exit(0)
    print('best epoch:{},aupr:{:.4f}, auc:{:.4f}'.format(best_epoch, best_aupr, best_auc))
    if para.out > 0:
        print("The best model is saved to {}".format(model_out_file))

    return auc_list









