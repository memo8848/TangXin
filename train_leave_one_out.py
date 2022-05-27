import sys
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from BDMF import BDMF

from DataLoad import DataLoad as dl
from DMF import DMF
from get_negative_samples import get_random_train
from get_negative_samples import get_reliable_train
from DrugTargetDataset import DrugTargetDataset as DTD
from Evaluate_hr_ndcg import evaluate_hr_ndcg
from WKNKN import WKNKN


#生成所有药(折)的测试数据
#在原矩阵inMat中抽100个没反应的target 再加上一个有反应的的target
# 返回 掩膜 W 测试列表test_data 维度为 num_drugs*101 测试标签test_label 格式[drug,target]
#问题：有些药物 掩膜后成全0了



def leava_one_out(intMat):
    seed=7
    num_drugs=intMat.shape[0]
    test_data=[]
    test_label=[]
    #每个药物抽有反应的对
    W = np.ones(intMat.shape)
    for i in range(num_drugs):
    #     二维数组平铺为一维数组 第i个药物所有正样本
        labels = np.argwhere(intMat[i] == 1).flatten()
    #   所有正样本里抽一个
        prng = np.random.RandomState(seed)
        index_p = prng.randint(len(labels))
    #     生成W矩阵
        W[i, labels[index_p]] = 0
    #     #生成test_label
        test_label.append([i,labels[index_p]])
    #生成test_data 在负样本里抽前100个
    for i in range(num_drugs):
        negatives = np.argwhere(intMat[i] == 0).flatten()
        negatives_list=np.append(negatives[:100],(test_label[i][1]))
        test_data.append(negatives_list)
    # print("w_intMat",(W*intMat),"test_data", np.array(test_data))
    return W,test_data,test_label


def train_leave_one_out(para,data_dir,dataset,num_negative,topK=10,k=0,kn=0,n=0):
    t1 = time()
    #拿到原始数据
    data = dl(data_dir, dataset)
    intMat = data.drug_target_matrix
    drugMat = data.drug_similarity
    targetMat = data.target_similarity
    num_drugs, num_targets = intMat.shape[0], intMat.shape[1]
    t2 = time()
    print("header:data, load time:{:.1f}, drugs:{:d}, targets:{:d}"
          .format(t2 - t1, num_drugs, num_targets))

    # 拿到训练和验证数据
    (W, test_data, test_label) = leava_one_out(intMat)
    if k == 0:
        train=get_random_train(num_negative,W * intMat)
    else:
        train = train=get_reliable_train(k, W * intMat, drugMat, targetMat)
    train = DTD(train[0], train[1], train[2], W * intMat)
    train = DataLoader(train, batch_size=para.batch_size, shuffle=True)


    #构建模型
    model = DMF(num_drugs, num_targets, para.layers)
    # model = BDMF(num_drugs, num_targets, para.layers)
    model.weight_init()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.reg)
    model_name = str(model.__class__)[17:][:-2]
    loss_name = str(criterion.__class__)[17 + 13:][:-2]
    print(
        'header:class:{},dataset:{}, batch_size:{}, epochs:{}, latent_dim:{}, num_negative:{}, lr:{}, reg:{},loss:{}'
            .format(model_name, dataset, para.batch_size, para.epochs, para.latent_dim, num_negative, para.lr, para.reg,
                    loss_name))

    #评估空白对照
    t1 = time()
    (hr, ndcg) = evaluate_hr_ndcg(model,W * intMat, test_data, test_label)
    t2 = time()
    print('epoch:0,train_time:{:.1f}s, HR:{:.4f}, NDCG:{:.4f}, test_time:{:.1f}s'.format(t1 - t1, hr, ndcg, t2 - t1))
    model_out_file = 'pretrain/' + '/{}-{}-{}-{}-{}-lr_{}-HR_{:.4f}-NDCG_{:.4f}-epoch_{}.model'.format(para.method,
                                                                                                       dataset,
                                                                                                       para.latent_dim,
                                                                                                       para.layers,
                                                                                                       num_negative,
                                                                                                       para.lr,
                                                                                                       hr,
                                                                                                       ndcg,
                                                                                                       0)
    if para.out > 0:
        torch.save(model.state_dict(), model_out_file)


    #训练模型
    best_hr, best_ndcg, best_iter, best_epoch = 0, 0, -1, -1
    count = 0
    loss_list=[]
    for epoch in range(para.epochs):
        model.train()
        epoch = epoch + 1
        t1 = time()
        #填装训练数据
        if k == 0:
            train = get_random_train(num_negative, W * intMat)
        else:
            train = get_reliable_train(k, W * intMat, drugMat, targetMat)
        train = DTD(train[0], train[1], train[2], W * intMat)
        train = DataLoader(train, batch_size=para.batch_size, shuffle=True)

        for batch_idx, (drug, target, y) in enumerate(train):
            y_hat = model(drug, target)
            loss = criterion(y_hat, y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time()

        #测试模型  几个eopch测试一回


        model.eval()

        #一次测试一个topk
        # if epoch % para.verbose == 0:
        #     if kn == 0:
        #         (hr, ndcg) = evaluate_hr_ndcg(model,W * intMat, test_data, test_label, topK)
        #     else:
        #         ww_intMat = WKNKN(W * intMat,drugMat,targetMat,kn,n)
        #         (hr, ndcg) = evaluate_hr_ndcg(model,ww_intMat , test_data, test_label, topK)
        #     print('epoch:{},train_time:{:.1f}s, HR:{:.4f}, NDCG:{:.4f}, test_time:{:.1f}s, loss:{:.6f}'
        #           .format(epoch, t2 - t1, hr, ndcg, time() - t2, loss))
        #     loss_list.append(round(loss.item(),2))

         #一次测试10个topk
        if epoch % para.verbose == 0:

            for k in range(topK):
                if kn == 0:
                    print('topk',k+1)
                    (hr, ndcg) = evaluate_hr_ndcg(model, W * intMat, test_data, test_label, k+1)
                else:
                    ww_intMat = WKNKN(W * intMat, drugMat, targetMat, kn, n)
                    (hr, ndcg) = evaluate_hr_ndcg(model, ww_intMat, test_data, test_label, k+1)
                print('epoch:{},train_time:{:.1f}s, HR:{:.4f}, NDCG:{:.4f}, test_time:{:.1f}s, loss:{:.6f}'
                      .format(epoch, t2 - t1, hr, ndcg, time() - t2, loss))
                loss_list.append(round(loss.item(), 2))


















            if hr> best_hr :
                count = 0
                best_train_time, best_hr, best_ndcg, best_epoch, best_test_time = t2 - t1, hr, ndcg, epoch, time() - t2
                model_out_file = 'pretrain/' + '/{}-{}-{}-{}-{}-lr_{}-HR_{:.4f}-NDCG_{:.4f}-epoch_{}.model'.format(
                    para.method,
                    dataset,
                    para.latent_dim,
                    para.layers,
                    num_negative,
                    para.lr,
                    hr,
                    ndcg,
                    epoch)
                if para.out > 0:
                    torch.save(model.state_dict(), model_out_file)
            else:
                count += 1
            if count == 50:
                print('best epoch:{},HR:{:.4f}, NDCG:{:.4f}'.format(best_epoch, best_hr, best_ndcg))
                if para.out > 0:
                    print("The best model is saved to {}".format(model_out_file))
                return loss_list
                sys.exit(0)
    print('best epoch:{},HR:{:.4f}, NDCG:{:.4f}'.format(best_epoch, best_hr, best_ndcg))
    if para.out > 0:
        print("The best model is saved to {}".format(model_out_file))

    return loss_list

    











