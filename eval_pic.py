import re
from DMF import DMF
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from DataLoad import DataLoad as dl


import torch

#画roc曲线
#所有折平均完得到的 fpr tpr 传过来

color_dict = {
    'nr': '#2878B5', #蓝
     'gpcr': '#C82423',#红
     'ic': '#00A087CC',#绿
    'e': '#FEB24C',#黄

    'drug': '#5B8982',
    'drugone': '#CFE4C2',
    'target': '#E58E61',
    'targetone': '#F3BE95',

    'sparsity':'#F3D2AC' #暗黄

     }    #用于设置线条的备选颜色

def ro_curve(fpr, tpr, figure_file, method_name,dataset):
    roc_auc = auc(fpr, tpr)
    lw = 1
    plt.plot(fpr, tpr,color=color_dict[dataset],
         lw=lw, label= method_name + ' (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#0C0D10',lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(font="Times New Roman",size=18,weight="bold")
    plt.yticks(font="Times New Roman",size=18,weight="bold")
    fontsize = 12
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(figure_file + ".png")
    plt.show()
    return





