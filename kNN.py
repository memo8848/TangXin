import numpy as np
from DataLoad import DataLoad as dl
#药物方 证明是负样本，靶点方 证明也是负样本 ，才当作可靠负样本 函数格式为[ [靶点，靶点...],[靶点，靶点...]...]

#drug的 sim 和target比  target的sim 和drug比
def isReliable(drug,target,top_sim_drugs, top_sim_targets,intMat):
    t_score=0
    d_score=0
    for target in top_sim_targets:
        if intMat[drug][target] ==0:
            d_score=d_score-1
        else:
            d_score=d_score+1
    for drug in top_sim_drugs:
        if intMat[drug][target] ==0:
            t_score=t_score-1
        else:
            t_score=t_score+1
    if t_score < 0 and d_score <0:
        #是可靠负样本
        return 0
        #不是可靠负样本
    else:
        return 1


#k和num_negative 不一样 k是取几个相似性 num_negative是一个drug取几个可靠负样本
def knn(k,num_negative,w_intMat,drugMat,targetMat):
    num_drug = w_intMat.shape[0]
    negative_all = []
    for drug in range(num_drug):
        #该药物的负样本列表
        negatives=[]
        # 求所有值为0 元素的下标
        unlabels = np.argwhere(w_intMat[drug] == 0).flatten()
        # print('药物',drug,'的全部负样本',unlabels)
        #找该药物的前k个相似药物
        top_sim_drugs = np.flipud(np.argsort(drugMat[drug]))[1:k + 1]
        # print('药物',drug,'的top相似',top_sim_drugs)
        #找num个负样本 摇k个target 验证是不是 不是再摇
        for i in range(num_negative):
            #target 实际被抽中的
            target = unlabels[np.random.randint(len(unlabels))]
            # print('摇到靶点', target)
            top_sim_targets = np.flipud(np.argsort(targetMat[target]))[1:k + 1]
            # print('靶点',target,'的top相似',top_sim_targets)
            while isReliable(drug,target,top_sim_drugs,top_sim_targets,w_intMat)==1:
                # print('不可靠')
                target = unlabels[np.random.randint(len(unlabels))]
                # print('摇到靶点', target)
                top_sim_targets = np.flipud(np.argsort(targetMat[target]))[1:k + 1]
            negatives.append(target)
            # print('药物', drug, '的可靠负样本', negatives)
        negative_all.append(negatives)
    print(negative_all)
    return negative_all

# data=dl('data','nr')
# knn(10,3,data.drug_target_matrix,data.drug_similarity,data.target_similarity)





#普通knn
# def knn(k,w_intMat,drugMat,targetMat):
#     num_drug = w_intMat.shape[0]
#
#     #求所有值为0 元素的下标
#     y_list_unlabel=np.argwhere(w_intMat == 0)
#
#     #负样本的下标[[di,tj]   ] 算法结果
#     y_list_neigative=[]
#
#     #处理相似性矩阵
#     for item in y_list_unlabel:
#        #根据k值找前k个相似的药物的下标
#        s_drug_k_list=np.flipud(np.argsort(drugMat[item[0]]))[1:k+1]
#        s_target_k_list =np.flipud(np.argsort(targetMat[item[1]]))[1:k + 1]
#
#        #测试
#        # print(item)
#        # s_target_k_sim_list = targetMat[s_target_k_list[0]][item[1]]
#        # print('第'+str(item[0])+'药物前k个相似的药物的下标',s_drug_k_list)
#        # print('第' + str(item[1]) + '靶点前k个相似的靶点的下标', s_target_k_list)
#        # print('第' + str(item[0]) + '靶点第1个相似的值', s_target_k_sim_list)
#
#        tp=0
#        dp=0
#        tn=0
#        dn=0
#
#        #对于药物 是否负样本？
#        #print('药物:靶点 ', item)
#        for i in range(k):
#            if w_intMat[s_drug_k_list[i]][item[1]] ==1:
#                dp=dp+1
#            else:
#                dn=dn+1
#
#          #对于靶点 是否负样本
#        for i in range(k):
#            if w_intMat[item[0]][s_target_k_list[i]] ==1:
#                tp = tp + 1
#            else:
#                tn = tn + 1
#
#
#        if dp<=dn and tp<=tn:
#            y_list_neigative.append(item)
#
#     y_list_neigative=np.array(y_list_neigative)
#
#
#     negative_all=[]
#     for i in range(num_drug):
#         negatives=[]
#         for interaction in y_list_neigative:
#             if interaction[0]==i:
#                 negatives.append(interaction[1])
#                 np.delete(y_list_neigative,interaction)
#         negative_all.append(negatives)
#     # print('negatives ', negative_all)
#     return negative_all
#
#
# #加权knn
# def wknn(k,w_intMat,drugMat,targetMat):
#     num_drug = w_intMat.shape[0]
#     #求权值
#     weight=[]
#     for i in range(k):
#         weight.append(np.exp(-(i+1) ** 2 / (2 * k ** 2)))
#
#     #求所有值为0 元素的下标
#     y_list_unlabel=np.argwhere(w_intMat == 0)
#
#     #负样本的下标[[di,tj]   ] 算法结果
#     y_list_neigative=[]
#
#     #处理相似性矩阵
#     for item in y_list_unlabel:
#        s_drug_k_list=np.flipud(np.argsort(drugMat[item[0]]))[1:k+1]
#        s_target_k_list =np.flipud(np.argsort(targetMat[item[1]]))[1:k + 1]
#
#        # 测试
#        # print(item)
#        # s_target_k_sim_list = targetMat[s_target_k_list[0]][item[1]]
#        # print('第'+str(item[0])+'药物前k个相似的药物的下标',s_drug_k_list)
#        # print('第' + str(item[1]) + '靶点前k个相似的靶点的下标', s_target_k_list)
#        # print('第' + str(item[0]) + '靶点第1个相似的值', s_target_k_sim_list)
#
#        tp =0
#        dp =0
#        for i in range(k):
#            if w_intMat[s_drug_k_list[i]][item[1]] ==1:
#                dp=dp+weight[i]
#            else:
#                dp=dp-weight[i]
#
#        for i in range(k):
#            if w_intMat[item[0]][s_target_k_list[i]] ==1:
#                tp = tp + weight[i]
#            else:
#                tp = tp - weight[i]
#
#        if dp<0 and tp<0:
#            y_list_neigative.append(item)
#
#     y_list_neigative=np.array(y_list_neigative)
#
#     negative_all = []
#     for i in range(num_drug):
#         negatives = []
#         for interaction in y_list_neigative:
#             if interaction[0] == i:
#                 negatives.append(interaction[1])
#                 np.delete(y_list_neigative, interaction)
#         negative_all.append(negatives)
#     # print('negatives ', negative_all)
#     return negative_all
#
#
