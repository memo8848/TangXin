# 标准库
import getopt
import sys

from train_k_flod import train_k_flod
from train_leave_one_out import train_leave_one_out
from model_remove import model_remove
#损失函数画图
from draw_pic import draw_line


#实验方法
#dmf+k_flod_cv1 dmf+k_flod_cv2 dmf+k_flod_cv3
# dmf+leave_one_out

#knn+dmf+k_flod_cv1 knn+dmf+k_flod_cv2 knn+dmf+k_flod_cv3
# knn+dmf+leave_one_out

#knn+dmf+wknkn+k_flod_cv1 #knn+dmf+wknkn+k_flod_cv2 #knn+dmf+wknkn+k_flod_cv3
# knn+dmf+wknkn+leave_one_out

def main(argv):
    # 测试设置

    #数据库
    data_dir = "data"
    dataset = 'nr'
    method = 0
    cvs = 3# cvs=1 去掉drug-target  cvs=2 去掉drug  cvs=3 去掉target

    #knn
    k = 6  #k=0代表不用knn

    #wknkn
    kn=5 #选取最近kn个已知邻居 kn=0代表不用wknkn
    n=0.6 #衰减系数

    #DMF
    epochs = 100
    batch_size = 64

    #要探究的参数
    layers = [128]


    latent_dim=layers[0]
    reg = 0.0#每层的l2正则化 0.25 一下 0.1可以  设置个search  0.125 0.25 0.5 1 看一下  e可以大一点
    num_negative = 1 #一个正样本配几个负样本
    lr = 0.0001  #lr设置0.0005 还可以 可以0.0005 0.0001 0.001 试一下  0.0001比较好
    learner = 'adam' #优化器 可选择  adagrad, adam, rmsprop, sgd
    verbose = 1 #verbose个 interation后 打一条log
    out = 0 #是否保存模型


    #Evaluate

    #leave_one_out
    topK = 10


    # 选择需要实验的方法
    methods = {0: "dmf_k", 1: "dmf_n", 2: "knn+dmf_k", 3: "knn+dmf_n", 4: "wknkn+dmf_1", 5: "wknkn+dmf_k"}

    opts, args = getopt.getopt(sys.argv[1:],'', [ "method=","data-dir=","dataset" 
                                                     "kNN=",
                                                     "epochs=", "batch_size=", "num_factors=", "layers=",
                                                     "reg=","num_negative=", "lr=", "learner=",
                                                     "verbose=", "out=", "regemlp_pretrain=",
                                                      "cvs="
                                                     ])
    for opt_name, opt_value in opts:
        if opt_name == '--method':
            data_dir = opt_value
            print("[*] method ", methods[int(opt_value)])
        #数据来源
        if opt_name =='--data-dir':
            data_dir = opt_value
            print("[*] data-dir ", data_dir)
        if opt_name =='--dataset':
            dataset = opt_value
            print("[*] dataset is ", dataset)
        #knn参数
        if opt_name == '--kNN':
            k = int(opt_value)
            print("[*] use kNN k=   ", k)
        #DMF参数
        if opt_name == '--epochs':
            epochs = int(opt_value)
            print("[*] epochs k=   ", epochs)

        if opt_name == "--batch_size":
            batch_size = int(opt_value)
            print("[*] batch_size=   ", batch_size)

        if opt_name == "--num_factors":
            num_factors = int(opt_value)
            print("[*] num_factors=   ", num_factors)

        if opt_name == "--layers":
            layers = list(map(int, opt_value.split(",")))
            print("[*] layers=   ", layers)

        if opt_name == "--reg":
            reg = float(opt_value)
            print("[*] regularization of each layer=   ", reg)

        if opt_name == "--num_negative":
            num_neg = int(num_negative)
            print("[*] number of negative=   ", num_negative)

        if opt_name == "--lr":
            lr = float(opt_value)
            print("[*] learning rate=   ", lr)

        if opt_name == "--learner":
            learner = opt_value
            print("[*] learner =   ", learner)

        if opt_name == "--verbose":
            verbose = int(opt_value)
            print("[*] numbers of iteration=   ", verbose)

        if opt_name == "--out":
            out = int(opt_value)
            print("[*] whether save the model=   ", out)

        if opt_name ==  "--regemlp_pretrain":
            regemlp_pretrain = opt_value
            print("[*] whether employ the pretrain models=   ", regemlp_pretrain)
        #验证参数
        if opt_name ==  "--cvs":
            cvs = int(opt_value)
            print("[*] type of cross varification=   ", cvs)


    # 网络参数
    class parameters:
        def __init__(self,method,epochs,batch_size,lr,layers,latent_dim,reg,verbose,out):
            self.method=method
            self.epochs=epochs
            self.batch_size=batch_size
            self.lr=lr
            self.layers=layers
            self.latent_dim=latent_dim
            self.reg=reg
            self.out=out
            self.verbose=verbose

    para=parameters(methods[method],epochs,batch_size,lr,layers,latent_dim,reg,verbose,out)


    # for i in range(7):
    #     k=i
    #     train_k_flod(para, data_dir, dataset, cvs, num_negative, k=k, kn=0, n=0)


    #画图loss 注意要是统一epoch 关掉早停
    # dataset_list=["nr","gpcr","ic","e"]
    # for dataset in dataset_list:
    #     train_k_flod(para,data_dir,dataset,cvs,num_negative,k=0,kn=0,n=0)





    # train_leave_one_out(para, data_dir, dataset, num_negative, topK=5, k=0, kn=0, n=0)




    # #实验
    print("applying method", methods[method])
    if method==0: #dmf_k
        print("cv", cvs)
        train_k_flod(para,data_dir,dataset,cvs,num_negative,k=0,kn=kn,n=n)

    # #     # model_remove("pretrain")
    # #     # model_remove("AUC")
    # if method==1:#dmf_n
    #     train_leave_one_out(para,data_dir,dataset,num_negative,topK=10,k=0,kn=0,n=0)
    #     model_remove("pretrain")
    #
    #
    # if method==2:#knn+dmf_k
    #     print("cv", cvs)
    #     train_k_flod(para,data_dir,dataset,cvs,num_negative,k=k,kn=0,n=0)
    #     # model_remove("pretrain")
    #     # model_remove("AUC")
    #
    #
    # if method==3:#knn+dmf_n
    #     train_leave_one_out(para,data_dir,dataset,num_negative,topK=10,k=k,kn=0,n=0)
    #     # model_remove("pretrain")
    #
    # if method == 4:  # dmf_k+wknkn
    #     print("cv", cvs)
    #     train_k_flod(para, data_dir, dataset, cvs, num_negative, k=0, kn=5, n=0.6)
    #     # model_remove("pretrain")
    #     # model_remove("AUC")
    #
    # if method == 5:  # dmf_n+wknkn
    #     train_leave_one_out(para, data_dir, dataset, num_negative, topK=10, k=0, kn=5, n=0.6)
    #     # model_remove("pretrain")
    #
    #
    # if method==6:#knn+dmf_k+wknkn
    #     print("cv", cvs)
    #     train_k_flod(para,data_dir,dataset,cvs,num_negative,k=0,kn=5,n=0.6)
    #     model_remove("pretrain")
    #     model_remove("AUC")
    #
    # if method==7:#knn+dmf_n+wknkn
    #     train_leave_one_out(para,data_dir,dataset,num_negative,topK=10,k=0,kn=5,n=0.6)
    #     model_remove("pretrain")



if __name__ == "__main__":
    main(sys.argv[1:])

