# 标准库
import argparse
from time import time
import sys

# 第三方库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 自定义库
from evaluate import evaluate_model
import utils
from dataset import Dataset

# 是否激活cuda
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print("torch.backends.cudnn.deterministic",torch.backends.cudnn.deterministic)
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#随机梯度下降
#小批梯度下降

#batchsize 小批梯度下降里的批大小
#iteration 使用大小为batchsize的样本过一遍
#epoch 整个训练集中样本过一遍


#优化器里的SGD实际上是小批梯度下降


#模型
class DMF(nn.Module):
    def __init__(self, num_users, num_items, layers, dataset): #矩阵行数，矩阵列数，网络层结构[输入，输出（输入），输出...]，数据集
        super(DMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        #隐变量维数 可能是m*n分成 m*r  和r*n  r就是这个
        #隐变量维数是layers[0]=64
        self.latent_dim = layers[0]
        #layers=[64,64]
        self.layers = layers


        self.user_item_indices = torch.LongTensor(np.array([dataset.user_indices, dataset.item_incides]))
        #[[用户下标的一维数组],[电影下标的一维数组]]
        self.rating_data = torch.FloatTensor(dataset.rating_data)
        #输入的数据
        # indices：此参数是指定非零元素所在的位置，也就是行和列，所以此参数应该是一个二维的数组，当然它可以是很多格式
        # 第一维指定了所有非零数所在的行数，第二维指定了所有非零元素所在的列数。例如indices = [[1, 4, 6], [3, 6, 7]]
        # 表示我们稀疏矩阵中(1, 3), (4, 6), (6, 7)几个位置是非零的数所在的位置。
        # values：此参数指定了非零元素的值，所以此矩阵长度应该和上面的indices一样长 values = [1, 4, 5]，表示上面的三个位置非零数分别为1, 4, 5。
        # size：指定了稀疏矩阵的大小，例如size = [10, 10]
        # 表示矩阵大小为10 × 10，如果不给定此值，那么默认是生成足以覆盖所有非零值的最小矩阵大小。
        # dtype：指定返回tensor中数据的类型，如果不指定，那么采取values中数据的类型。
        # device：指定创建的tensor在cpu还是cuda上。
        # requires_grad：指定创建的tensor需不需要梯度信息，默认为False

        self.user_item_matrix = torch.sparse_coo_tensor(self.user_item_indices, self.rating_data,
                                                        torch.Size((self.num_users, self.num_items))).to_dense().to(device) #todense将压缩存储格式转化为正常矩阵的样子




        #网络结构 这是两个一样的网络啊 一个光处理行，一个光处理列
        #构造输入层
        self.linear_user_1 = nn.Linear(in_features=self.num_items, out_features=self.latent_dim)
        #detach()截断反向传播梯度流，可能把是因为第一层 没有处理，没必要求梯度
        self.linear_user_1.weight.detach().normal_(0, 0.01)
        self.linear_item_1 = nn.Linear(in_features=self.num_users, out_features=self.latent_dim)
        self.linear_item_1.weight.detach().normal_(0, 0.01)
        #构造中间层，根据lays构造若干个全连接层
        self.user_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):#对网络每一层
            self.user_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))#输入是前一层大小，输出是这一层大小

        self.item_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.item_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))


#前向传播
    def forward(self, user_indices, item_indices):
        #准备输入数据
        user = self.user_item_matrix[user_indices]
        item = self.user_item_matrix[:, item_indices].t()
        #输入层
        user = self.linear_user_1(user)
        item = self.linear_item_1(item)
        #中间全连接层
        for idx in range(len(self.layers) - 1):
            user = F.relu(user)
            user = self.user_fc_layers[idx](user)

        for idx in range(len(self.layers) - 1):
            item = F.relu(item)
            item = self.item_fc_layers[idx](item)
        #计算损失函数 余弦相似性？
        vector = torch.cosine_similarity(user, item).view(-1, 1)
        #把向量值压缩到0（1*e-6）和1之间
        vector = torch.clamp(vector, min=1e-6, max=1)
        #返回前向传播结果
        return vector

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Conv1.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layers', nargs='?', default='[64,64]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg', type=float, default='0.0',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--emlp_pretrain', nargs='?', default='1',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


if __name__ == '__main__':
    print("NAME",__name__)

    # settings
    args = parse_args()
    path = args.path
    dataset_name = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    #eval函数功能：将字符串str当成有效的表达式来求值并返回计算结果。eval函数可以实现list、dict、tuple与str之间的转化
    layers = eval(args.layers)
    latent_dim = layers[0]
    reg = args.reg
    learning_rate = args.lr
    num_negative = args.num_neg
    verbose = args.verbose
    out = args.out
    emlp_pretrain = args.emlp_pretrain
    #
    topK = 10
    #
    evaluation_threads = 1  # mp.cpu_count()

    # Loading data
    t1 = time()
    dataset = Dataset(path + dataset_name)
    train, test_rating, test_negative = dataset.get_train_instances(num_negative), dataset.test_ratings, dataset.test_negative
    num_users, num_items = dataset.num_users, dataset.num_items
    t2 = time()
    print("header:data, load time:{:.1f}, user:{:d},train:{:d} item:{:d}, test:{:d}"
          .format(t2 - t1, num_users, len(train[0]), num_items, len(test_rating)))

    train = utils.UserItemRatingDataset(train[0], train[1], train[2])
    train = DataLoader(train, batch_size=batch_size, shuffle=True)


    #测试加载数据的格式
    print("测试数据加载格式")
    model = DMF(num_users, num_items, layers, dataset)
    model = model.to(device)
    # 测试加载数据的格式
    print("测试数据加载格式")
    print(" DMF.user_item_matrix中行 users", model.user_item_matrix[0])
    print(" DMF.user_item_matrix中列 items", model.user_item_matrix[:, 0])

    # # Build model
    # model = DMF(num_users, num_items, layers, dataset)
    # model = model.to(device)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
    # print(model)
    #
    #
    # model_name = str(model.__class__)[17:][:-2]
    # loss_name = str(criterion.__class__)[17 + 13:][:-2]
    # print(
    #     'header:class:{},dataset:{}, batch_size:{}, epochs:{}, latent_dim:{}, num_negative:{}, topK:{}, lr:{}, reg:{},loss:{}'
    #     .format(model_name, dataset_name, batch_size, epochs, latent_dim, num_negative, topK, learning_rate, reg,
    #             loss_name))
    #
    # # Init performance
    # t1 = time()
    # (hr, ndcg) = evaluate_model(model, test_rating, test_negative, topK, evaluation_threads)
    # t2 = time()
    # print('epoch:0,train_time:{:.1f}s, HR:{:.4f}, NDCG:{:.4f}, test_time:{:.1f}s'.format(t1 - t1, hr, ndcg, t2 - t1))
    # model_out_file = 'pretrain/'+'/{}-{}-{}-{}-{}-lr_{}-HR_{:.4f}-NDCG_{:.4f}-epoch_{}.model'.format(model_name,
    #                                                                                                     dataset_name,
    #                                                                                                     latent_dim,
    #                                                                                                     layers,
    #                                                                                                     num_negative,
    #                                                                                                     learning_rate,
    #                                                                                                     hr,
    #                                                                                                     ndcg,
    #                                                                                                     0)
    # if args.out > 0:
    #     torch.save(model.state_dict(), model_out_file)
    #
    # # Train model
    # best_hr, best_ndcg, best_iter, best_epoch = 0, 0, -1, -1
    # count = 0
    # for epoch in range(epochs):
    #     model.train()
    #     epoch = epoch + 1
    #     t1 = time()
    #     # Generate training instances
    #     train = dataset.get_train_instances(num_negative)
    #     train = utils.UserItemRatingDataset(train[0], train[1], train[2])
    #     train = DataLoader(train, batch_size=batch_size, shuffle=True)
    #     # Training
    #     for batch_idx, (user, item, y) in enumerate(train):
    #        # user, item, y = user.cuda(), item.cuda(), y.cuda()
    #         ## forward and backprop
    #         y_hat = model(user, item)
    #         loss = criterion(y_hat, y.view(-1, 1))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     t2 = time()
    #     model.eval()
    #
    #     # Evaluation
    #     if epoch % verbose == 0:
    #         (hr, ndcg) = evaluate_model(model, test_rating, test_negative, topK, evaluation_threads)
    #         print('epoch:{},train_time:{:.1f}s, HR:{:.4f}, NDCG:{:.4f}, test_time:{:.1f}s, loss:{:.6f}'
    #               .format(epoch, t2 - t1, hr, ndcg, time() - t2, loss))
    #         if hr > best_hr:
    #             count = 0
    #             best_train_time, best_hr, best_ndcg, best_epoch, best_test_time = t2 - t1, hr, ndcg, epoch, time() - t2
    #             model_out_file = 'pretrain/' + '/{}-{}-{}-{}-{}-lr_{}-HR_{:.4f}-NDCG_{:.4f}-epoch_{}.model'.format(
    #                 model_name,
    #                 dataset_name,
    #                 latent_dim,
    #                 layers,
    #                 num_negative,
    #                 learning_rate,
    #                 hr,
    #                 ndcg,
    #                 epoch)
    #             if args.out > 0:
    #                 torch.save(model.state_dict(), model_out_file)
    #         else:
    #             count += 1
    #         if count == 50:
    #             sys.exit(0)
    #
    # print('best epoch:{},HR:{:.4f}, NDCG:{:.4f}'.format(best_epoch, best_hr, best_ndcg))
    # if args.out > 0:
    #     print("The best model is saved to {}".format(model_out_file))


    #验证
    print("测试")
    #print(test(torch.tensor([0., 0., 0.])))