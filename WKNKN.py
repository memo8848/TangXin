import numpy as np
from DataLoad import DataLoad as dl


def WKNKN(intMat,drugMat,targetMat,kn,n):
    # print("矩阵w_intMat",intMat)
    num_drug=intMat.shape[0]
    num_target=intMat.shape[1]
    #初始化权值矩阵
    Wd=np.zeros(intMat.shape)
    Wt = np.zeros(intMat.shape)
    for drug in range(num_drug):
        # print("药物",drug)
        nearest_drugs=[]
        # 找所有从远到近排序(包括本身)
        nearer= np.argsort(drugMat[drug])
        #存储着原始的 从近到远的顺序(包括本身)
        nearer_dict=np.flipud(nearer)[1:].tolist()
        # print("邻居们由近及远", nearer_dict)
        #转成列表 方便求最近邻居
        nearer_drugs=nearer.tolist()
        #去掉本身
        nearer_drugs.pop()
        # print("邻居们由远及近", nearer_drugs)
        #判断是否已知 是了 加入列表
        for k in range(kn):
            nearest=nearer_drugs.pop()
            # print("最近邻居", nearest)
            while (np.all(intMat[nearest] ==0)):
                # print("最近邻居全0")
                try:
                    nearest = nearer_drugs.pop()
                except IndexError:
                    break
                # print("新最近邻居",nearest)
            nearest_drugs.append(nearest)
        # print("已知最近邻居们由近及远", nearest_drugs)
        # print('nearest_drugs',nearest_drugs)
        #k个权值 t
        z=0
        for i in range(kn):
            #正则化项 最近已知邻居的相似度 求和
            z=z+drugMat[drug][nearest_drugs[i]]
        # print('zd1', z)
        # print('zd2', 1/z)
        #该药物的打分行
        for i in range(kn):
            #求nearest_drugs[i]在实际的nearer_dict中的位置
            # print("表面名次", i)
            # print("实际名次", nearer_dict.index(nearest_drugs[i]))
            Wd[drug]=Wd[drug]+pow(n,nearer_dict.index(nearest_drugs[i])+1)*intMat[nearest_drugs[i]]
        Wd[drug]=Wd[drug]*(1/z)
    # print('Wd', Wd)

    for target in range(num_target):
        # 找所有从远到近排序
        # print("靶点", target)
        nearest_targets = []
        nearer=np.argsort(targetMat[target])
        nearer_dict=np.flipud(nearer)[1:].tolist()
        nearer_targets = nearer.tolist()
        #去掉本身
        nearer_targets.pop()
        # print("邻居们由远及近", nearer_targets)
        # 判断是否已知 是了 加入列表
        for k in range(kn):
            nearest = nearer_targets.pop()
            # print("最近邻居", nearest)
            while (np.all(intMat[:nearest] == 0)):
                # print("最近邻居全0")
                try:
                    nearest = nearer_targets.pop()
                except IndexError:
                    # print("最近邻居列表", nearer_targets)
                    # print("下标越界,当前列表",nearest_targets)
                    break
                # print("新最近邻居",nearest)
            nearest_targets.append(nearest)
        # print("已知最近邻居们由近及远", nearest_targets)
        # print('nearest_targets',nearest_targets)
        # k个权值 t
        z=0
        for i in range(kn):
            #归一化项
            z=z+targetMat[target][nearest_targets[i]]
        # print('z1', z)
        # print('z2', 1/z)
        for i in range(kn):
            # print("表面名次", i)
            # print("实际名次", nearer_dict.index(nearest_targets[i]))
            Wt[:,target]=Wt[:,target]+pow(n,nearer_dict.index(nearest_targets[i])+1)*intMat[:,nearest_targets[i]]
        Wt[:,target]=Wt[:,target]*(1/z)
    # print('Wt', Wt)
    W=0.5*(Wd+Wt)
    #取W和原矩阵中较大值
    max_index = np.argwhere(intMat == 1)
    for (d,t) in max_index:
        W[d][t]=1
    # print('W', W)
    return W

# #测试
# data_dir='data'
# dataset = 'nr'
# kn=5
# n=0.6
# data = dl(data_dir,dataset)
# intMat=data.drug_target_matrix
# drugMat=data.drug_similarity
# targetMat=data.target_similarity
# (W,test_data,test_label)=leava_one_out(intMat)
# WKNKN(W * intMat,drugMat,targetMat,kn,n)
