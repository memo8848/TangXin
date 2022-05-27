import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoad import DataLoad as dl
import os
import xlrd
import pandas as pd


color_dict = {
    'nr': '#108890', #绿
     'gpcr': '#EA5E51',#红
     'ic': '#4699C2',#蓝
    'e': '#F2933D',#黄

    'dmf': '#4699C2',
    'wknkn':'#EA5E51',
    'cv1': '#108890',
    'cv2': '#4699C2',
    'cv3':'#FFD574'

     }    #用于设置线条的备选颜色

font= {
     'family': 'serif',
     'weight': 'bold',
     'size': '8',
}

def draw_bar():
    data_dir='data'
    datasets=['nr','gpcr','ic','e']
    save_dict = 'evaluate/'

    num_drug_list=[]
    one_drug_list = []

    num_target_list=[]
    one_target_list = []

    sparsity_list=[]
    for d in datasets:
        print('数据集',d)
        data=dl(data_dir,d)
        intMat=data.drug_target_matrix
        num_drugs=data.num_drugs
        print('药物数', num_drugs)
        num_targets=data.num_targets

        num_drug_list.append(num_drugs)
        num_target_list.append(num_targets)

        one_interaction_drug=0
        one_interaction_target =0
        for item in intMat:
            if len(np.argwhere(item == 1))==1:
                one_interaction_drug=one_interaction_drug+1
        print('一个靶点数', one_interaction_drug)
        one_drug_list.append(one_interaction_drug)
        for item in intMat.T:
            if len(np.argwhere(item == 1))==1:
                one_interaction_target=one_interaction_target+1
        print('一个药物数', one_interaction_target)
        one_target_list.append(one_interaction_target)

        num_interactions = len(np.argwhere(intMat == 1))
        print('作用对个数', num_interactions)
        sparsity = num_interactions / (num_drugs * num_targets)
        print('稀疏度', sparsity)
        sparsity_list.append(sparsity)

    # print('药物数表', num_drug_list)
    # print('一个药物数表', one_drug_list)
    # print('靶点数表', num_target_list)
    # print('一个靶点数', one_target_list)
    # print('稀疏度表', sparsity_list)

    width= 0.35
    x=list(range(len(datasets)))
    xd = list(range(len(datasets)))
    xt = list(range(len(datasets)))
    plt.grid(axis='y', linestyle='--')

    d1 = plt.bar(xd, num_drug_list, label='drug', fc=color_dict['drug'],width=width)
    d2 = plt.bar(xd,  one_drug_list, bottom=num_drug_list, label='one drug', tick_label=datasets, fc=color_dict['drugone'],width=width)
    for i in range(len(datasets)):
        xd[i]=x[i]+width
    t1 = plt.bar(xd, num_target_list, label='target', fc=color_dict['target'],width=width)
    d2 = plt.bar(xd, one_target_list, bottom=num_target_list, label='one target', tick_label=datasets, fc=color_dict['targetone'],width=width)

    for i in range(len(datasets)):
        xt[i] = x[i] + width/2
    plt.xticks(xt,datasets)
    ax = plt.gca()  # 返回坐标轴
    ax.tick_params(axis='x', tickdir='in')
    ax.tick_params(axis='y', tickdir='out')

    # 显示图例
    plt.legend(loc='upper left', prop=font, handlelength=3.5)
    plt.savefig(save_dict + 'dataset', dpi=300, bbox_inches='tight')  # , dpi=100
    plt.show()


    #第二张图
    yticks=list(np.arange(start=0, stop=0.1, step=0.01))
    labels=[]
    for y in yticks:
        labels.append('{:.2f}%'.format(y*100))

    ax = plt.gca()  # 返回坐标轴
    ax.tick_params(axis='x', tickdir='in')
    ax.tick_params(axis='y', tickdir='out')

    plt.xticks(x, datasets)
    plt.yticks(yticks, labels)
    plt.grid(axis='y', linestyle='--')
    plt.bar(x,sparsity_list,fc=color_dict['sparsity'],width=width, label='sparsity')
    plt.legend(loc='upper right', prop=font, handlelength=3.5)

    plt.savefig(save_dict+'sparsity', dpi=300, bbox_inches='tight')  #, dpi=100
    plt.show()


# 图的存储名：模型名+aupr 模型名+auc 图形类型(1:aupr 2:auc) x值(1:  2:) y值(1:  2:  )
def draw_top_k(save_dir, file_1,file_2):
    cl = color_dict
    workbook_1 = xlrd.open_workbook(save_dir + file_1 + '.xls')
    workbook_2 = xlrd.open_workbook(save_dir + file_2 + '.xls')
    sheet_names = workbook_1.sheet_names()

    #dmf 四个数据库
    hr_dict_1 = {}
    hr_dict_2 = {}

    #wknkn 四个数据库
    ncdg_dict_1={}
    ncdg_dict_2 = {}
    for dataset in sheet_names:
        train_list_1 = workbook_1.sheet_by_name(dataset)
        train_list_2 = workbook_2.sheet_by_name(dataset)

        hr_1 = []
        ncdg_1 = []
        hr_2 = []
        ncdg_2 = []
        topk = train_list_1.nrows
        for idx in range(topk):
            row_1 = train_list_1.row_values(idx)[0]
            row_2 = train_list_2.row_values(idx)[0]

            #对于dmf
            hr_1.append(float(str(row_1).split(':')[1].split(',')[0]))
            # print('hr',hr_1)
            ncdg_1.append(float(str(row_1).split(':')[2]))
            # print('ncdg', ncdg_1)

            #对于wknkn
            hr_2.append(float(str(row_2).split(':')[1].split(',')[0]))
            ncdg_2.append(float(str(row_2).split(':')[2]))


        hr_dict_1[dataset] = hr_1
        ncdg_dict_1[dataset]=ncdg_1

        hr_dict_2[dataset] = hr_2
        ncdg_dict_2[dataset] = ncdg_2

    # 处理y值 统一加0.1
    for dataset in sheet_names:
        for a in range(topk):
            # print(hr_dict_2[dataset][a])
            hr_dict_1[dataset][a]=float(hr_dict_1[dataset][a])+ float(0.1)
            ncdg_dict_1[dataset][a]=float(ncdg_dict_1[dataset][a])+ float(0.1)


            hr_dict_2[dataset][a]=float(hr_dict_2[dataset][a])+ float(0.1)
            ncdg_dict_2[dataset][a]=float(ncdg_dict_2[dataset][a])+ float(0.1)





            #dmf
    print('dmf')
    print(hr_dict_1)
    print(ncdg_dict_1)
    #wknkn
    print('wknkn')
    print(hr_dict_2)
    print(ncdg_dict_2)

    #一个数据库画俩张图 一个hr 一个NDCG  两条线 一个dmf 一个WKNKN
    for dataset in sheet_names:
        # #设置横纵坐标的刻度范围
        plt.xlim((1, topk))  # x轴的刻度范围被设为a到b
        plt.ylim((0.2, 0.9))  # x轴的刻度范围被设为a到b

        plt.xlabel('N', fontdict=font, fontsize=12)
        plt.ylabel('HR@N', fontdict=font, fontsize=12)

        # 设置y轴的显示刻度线的几个值
        ytick = np.arange(start=0.2, stop=0.9, step=0.05)
        plt.yticks(ytick, fontsize=8)

        xtick = np.arange(start=1, stop=topk, step=1)
        plt.xticks(xtick, fontsize=10)  # 对于X轴，只显示x中各个数对应的刻度值

        ax = plt.gca()  # 返回坐标轴
        ax.tick_params(axis='x', tickdir='in')
        ax.tick_params(axis='y', tickdir='out')

        # 设置网格
        plt.grid(axis='y', linestyle='--')

        # x值都一样
        x = np.arange(start=1, stop=topk + 1, step=1)





        # 画图1
        plt.plot(x, hr_dict_1[dataset], label='dmf', marker='s',linestyle='--',mfc=cl['dmf'], c=cl['dmf'], linewidth=0.75, ms=6)
        plt.plot(x, hr_dict_2[dataset],label='wknkn+dmf', marker='o', mfc=cl['wknkn'], c=cl['wknkn'], linewidth=0.75, ms=6)
        # 显示图例
        ax.legend(loc='lower right', ncol=2, prop=font, handlelength=3.5)
        # 保存图像
        plt.savefig(save_dir + 'hr_' + dataset + '.png', dpi=300, bbox_inches='tight')
        plt.show()










        # 画图2 ndcg

        plt.xlim((1, topk))  # x轴的刻度范围被设为a到b
        plt.ylim((0.2, 0.9))  # x轴的刻度范围被设为a到b

        plt.xlabel('N', fontdict=font, fontsize=12)
        plt.ylabel('NDCG@N', fontdict=font, fontsize=12)


        # 设置y轴的显示刻度线的几个值
        ytick = np.arange(start=0.2, stop=0.9, step=0.05)
        plt.yticks(ytick, fontsize=8)

        xtick = np.arange(start=1, stop=topk, step=1)
        plt.xticks(xtick, fontsize=10)  # 对于X轴，只显示x中各个数对应的刻度值

        ax = plt.gca()  # 返回坐标轴
        ax.tick_params(axis='x', tickdir='in')
        ax.tick_params(axis='y', tickdir='out')

        # 设置网格
        plt.grid(axis='y',linestyle='--')

        # x值都一样
        x = np.arange(start=1, stop=topk + 1, step=1)


        plt.plot(x, ncdg_dict_1[dataset], label='dmf',  marker='s',linestyle='--',mfc=cl['dmf'], c=cl['dmf'], linewidth=0.75, ms=6)
        plt.plot(x, ncdg_dict_2[dataset], label='wknkn_dmf', marker='o', mfc=cl['wknkn'], c=cl['wknkn'], linewidth=0.75, ms=6)
        # 显示图例
        ax.legend(loc='lower right', ncol=2, prop=font, handlelength=3.5)
        # 保存图像
        print(save_dir)
        plt.savefig(save_dir +'ndcg_' + dataset+ '.png', dpi=300, bbox_inches='tight')
        plt.show()

# draw_top_k('HR/','dmf','wknkn')





#画折线图 训练中的AUC
def draw_line(file):
    save_dict = 'AUC/'
    cl = color_dict


    workbook = xlrd.open_workbook(save_dict+file + '.xls')
    sheet_names = workbook.sheet_names()
    auc_dict = {}
    for dataset in sheet_names:
        auc=[]
        train_list = workbook.sheet_by_name(dataset)
        epochs = train_list.nrows
        for idx in range(epochs):
           row = train_list.row_values(idx)[0]
           auc.append(float(str(row).split(':')[4].split(',')[0]))
        auc_dict[dataset]=auc
    # print(auc_dict)

    #画图
    x=np.arange(start=0,stop=epochs,step=1)

    for (key,val) in auc_dict.items():
        plt.plot(x,val,label=key,mfc=cl[key],c=cl[key], linewidth=1, ms=6)
    # #设置横纵坐标的刻度范围
    plt.xlim((0, epochs))   #x轴的刻度范围被设为a到b
    plt.ylim((0.7, 1) ) # x轴的刻度范围被设为a到b
    plt.xlabel('epochs',fontdict=font,fontsize=12)
    plt.ylabel('AUC',fontdict=font,fontsize=12)



    #设置y轴的显示刻度线的几个值

    ytick = np.arange(start=0.7, stop=1, step=0.05)
    plt.yticks(ytick, fontsize=10)

    xtick = np.arange(start=0, stop=epochs, step=epochs/10)
    plt.xticks(xtick,fontsize=10)   #对于X轴，只显示x中各个数对应的刻度值

    ax = plt.gca()  #返回坐标轴
    ax.tick_params(axis='x', tickdir='in')
    ax.tick_params(axis='y', tickdir='out')

    #设置网格
    plt.grid(linestyle='--')

    #显示图例
    ax.legend(loc='lower right',ncol=2,prop=font,  handlelength=3.5)


    #保存图像
    print(save_dict)
    plt.savefig(save_dict+file+'png', dpi=300, bbox_inches='tight')

    #显示图
    plt.show()


draw_line('epoch_auc')

color_dict = {
    'nr':'Oranges', #蓝
     'gpcr': 'RdPu',#红
     'ic':  'Blues',#绿
    'e': 'Greens',#黄

     }

#画热力图   给文件名 解析文件名
def draw_matrix(save_dir,file):
    # data=pd.read_excel(file+'.xls')
    workbook = xlrd.open_workbook(save_dir+file+'.xls')
    sheet_names = workbook.sheet_names()


    for dataset in sheet_names:
        data=[]
        matrix = workbook.sheet_by_name(dataset)
        row_num=matrix.nrows
        col_num=matrix.ncols
        for idx in range(row_num):
            data.append(matrix.row_values(idx))
        data=np.array(data)
        print(dataset)
        print(data)
        x_ticks=["1ce","2ce","3ce"]
        y_ticks = ["16","32", "64", "128"]
        plt.figure(num=1, figsize=(6, 6))
        sns.heatmap(data, annot=True,
                    square=True,
                    cmap=color_dict[dataset],
                    xticklabels=x_ticks,
                    yticklabels=y_ticks,
                    fmt='.3f'
                    )

        plt.savefig(save_dir + file+"_"+dataset, dpi=300, bbox_inches='tight')
        plt.show()

# draw_matrix("AUC/","dmf_auc ")






