# 第三方库
import torch
import torch.nn as nn
import torch.nn.functional as F

class BDMF(nn.Module):
    def __init__(self, num_drugs, num_targets, layers): #矩阵行数，矩阵列数，网络层结构[输入，输出（输入），输出...]
        super(BDMF, self).__init__()
        self.num_drugs = num_drugs
        self.num_targets = num_targets
        #隐变量维数 可能是m*n分成 m*r  和r*n  r就是这个

        #隐变量维数是layers[0]=64
        self.latent_dim = layers[0]
        self.layers = layers


        #网络结构 这是两个一样的网络 一个处理行，一个处理列
        #构造输入层
        self.linear_drug_1 = nn.Linear(in_features=self.num_targets, out_features=self.latent_dim)
        #detach()截断反向传播梯度流，可能把是因为第一层 没有处理，没必要求梯度
        self.linear_drug_1.weight.detach().normal_(0, 0.01)

        self.linear_target_1 = nn.Linear(in_features=self.num_drugs, out_features=self.latent_dim)
        self.linear_target_1.weight.detach().normal_(0, 0.01)

        #构造中间层，根据lays构造若干个全连接层
        self.drug_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):#对网络每一层
            self.drug_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))#输入是前一层大小，输出是这一层大小

        self.target_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.target_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))


    #bias层
        self.drug_bias_layer=nn.Linear(in_features=self.num_targets,out_features=self.latent_dim)
        self.target_bias_layer = nn.Linear(in_features=self.num_drugs, out_features=self.latent_dim)


    #前向传播
    def forward(self, drug, target):

        # # bias层 和输入层同时输入
        drug_bias = self.drug_bias_layer(drug)
        target_bias = self.target_bias_layer(target)


        #输入层
        drug = self.linear_drug_1(drug)
        target = self.linear_target_1(target)




        #中间全连接层
        for idx in range(len(self.layers) - 1):
            drug =F.relu(drug)
            drug = self.drug_fc_layers[idx](drug)
            # print('药物',drug)

        for idx in range(len(self.layers) - 1):
            target = F.relu(target)
            target = self.target_fc_layers[idx](target)
            # print('靶点',target)

        #furion 逐个相加
        drug = drug+drug_bias
        target = target+target_bias




        vector = torch.cosine_similarity(drug, target).view(-1, 1)
        # 把向量值压缩到0（1*e-6）和1之间
        vector = torch.clamp(vector, min=1e-6, max=1)
        # # print(vector)
        # #返回前向传播结果
        return vector


    def weight_init(self):
        nn.init.xavier_normal_(self.drug_bias_layer.weight)
        nn.init.constant_(self.drug_bias_layer.bias,0)
        nn.init.xavier_normal_(self.target_bias_layer.weight)
        nn.init.constant_(self.target_bias_layer.bias,0)

        for layer in self.drug_fc_layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

        for layer in self.target_fc_layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)




