
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from eval_pic import ro_curve



#验证一折
def eval_one_flod(model,cvs,test_data,test_label,w_intMat):
    drugs=[]
    targets=[]
    device = torch.device("cpu")
    # print(test_data,test_label)
    if cvs ==1 or cvs ==2:
        for d, t in test_data:
            drugs.append(w_intMat[d])
            targets.append(w_intMat[:,t].T)
    if cvs ==3:
        for t, d in test_data:
            targets.append(w_intMat[:,t].T)
            drugs.append(w_intMat[d])
    batch_drugs, batch_targets = torch.FloatTensor(np.array(drugs)), torch.FloatTensor(np.array(targets))
    tensor_drugs, tensor_targets = batch_drugs.to(device), batch_targets.to(device)
    # 输入模型进行预测
    y_pred = model(tensor_drugs, tensor_targets)
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()
    scores=y_pred.flatten()
    # print(" y_pred", scores)


    prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
    # print(" prec", np.array(prec))
    # print(" rec", np.array(prec))
    # print(" thr", np.array(prec))

    sklearn.metrics.precision_recall_curve(test_label, np.array(scores), pos_label=None, sample_weight=None)
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(test_label, np.array(scores))

    auc_val = auc(fpr, tpr)

    return fpr,tpr,aupr_val, auc_val
    # return aupr_val, auc_val

