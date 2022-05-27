import numpy as np
from DataLoad import  DataLoad as dl

def k_nearest(k,w_intMat,drugMat,targetMat):
    num_drugs=w_intMat.shape[0]
    num_targets=w_intMat.shape[1]
    intMat=w_intMat


    #设置0+ 就是1  反正我们只需要0-

    #存所有drug的k近邻 不要反复查了
    drugs_neighbors=[]
    for i,drug in enumerate(w_intMat):
        drugs_neighbors.append(np.flipud(np.argsort(drugMat[i]))[1:k+1])

    targets_neighbors = []
    for i,target in enumerate(w_intMat.T):
        targets_neighbors.append(np.flipud(np.argsort(targetMat[i]))[1:k + 1])


    for d in range(num_drugs):
        #和d反应的t 长度不定
        t_act=np.argwhere(w_intMat[d] == 1).flatten()
        # print('t_act',t_act)
        #和d类似的d 长度为k
        drug_sim=drugs_neighbors[d]
        for i in drug_sim:
            for j in t_act:
                #设置成0+ 就是1
                # print('set',i,j,'0+')
                intMat[i,j]=1


    for t in range(num_targets):
        #和t反应的d 长度不定
        d_act=np.argwhere(w_intMat[:,t] == 1).flatten()
        # print('d_act',d_act)
        #和t类似的t 长度为k
        target_sim=targets_neighbors[t]
        for i in d_act:
            for j in target_sim:
                #设置成0+ 就是1
                # print('set', i, j, '0+')
                intMat[i,j]=1


    # print(drugs_neighbors)
    # print(targets_neighbors)

    negative_all=np.argwhere(intMat == 0)
    # print(negative_all)
    print(negative_all.shape)
    return negative_all

# data=dl('data','nr')
# k_nearest(7,data.drug_target_matrix,data.drug_similarity,data.target_similarity)
