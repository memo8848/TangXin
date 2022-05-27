import os
#
# # 指定目录位置
# #功能 移走一样参数的前几个 留下最后一个  模型参数-学习率—结果评估  就是 前几个结果评估都删了 只留下同学习率最后一个结果评估

def model_remove(model_dir):

    # 遍历所有文件
    file_names = os.listdir(model_dir)
    file_list = []
    # 迭代每个文件名
    print('file_names',file_names)
    file_1 = file_names[0]
    file_1_split = file_1.split('_')
    print('file_names', file_1_split)
    for i, item in enumerate(file_names, 1):
        file_2 = item
        file_2_split = file_2.split('_')
        #如果参数设置不同
        if file_1_split[0:2] != file_2_split[0:2]:
            file_list.append(file_1)
        file_1 = file_2
        file_1_split = file_2_split
    file_list.append(file_1)
    for e in file_list:
        file_names.remove(e)
    print(len(file_names))
    for file in file_names:
        os.remove(model_dir+'\\'+file)
    print('delete')


# model_remove('pretrain')

# file_names=['a_1_1','a_1_2','a_1_3','b_1','b_2','b_3','c_1']
# file_list=[]
# print('file_names',file_names)
# file_1 = file_names[0]
# file_1_split = file_1.split('_')
# print('file_1',file_1)
# for i, item in enumerate(file_names, 1):
#     print('循环',i)
#     file_2 = item
#     print('file_2', file_2)
#     file_2_split = file_2.split('_')
#     print('file_2_split', file_2_split,'file_1_split', file_1_split)
#     #如果参数设置不同
#     if file_1_split[0:2] != file_2_split[0:2]:
#         print('file_2_split不等于file_1_split', file_2_split,  file_1_split)
#         #file_list加入file1
#
#         file_list.append(file_1)
#         print('file_list', file_list)
#     file_1 = file_2
#     file_1_split = file_2_split
# file_list.append(file_1)
# print('file_list',file_list)
# for e in file_list:
#     file_names.remove(e)
# print(len(file_names))
# print('delete')
