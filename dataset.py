import numpy as np
from scipy.sparse import csr_matrix
import os

class Dataset(object):

    def __init__(self, path):

        self.train_ratings, self.train_num_users, self.train_num_items = self.load_train_rating_file_as_list(path + ".train.rating")
        self.test_ratings, self.test_num_users, self.test_num_items = self.load_test_rating_file_as_list(path + ".test.rating")

        self.num_users = max(self.train_num_users, self.test_num_users)
        self.num_items = max(self.train_num_items, self.test_num_items)

        self.test_negative = self.load_negative_file(path + ".test.negative")

        self.user_item_rating_indices = self.get_user_item_matrix_indices()
        self.user_indices, self.item_incides, self.rating_data = self.user_item_rating_indices

        self.user_matrix = self.get_user_sparse_matrix()
        self.item_matrix=self.get_item_sparse_matrix()




        # print("用户数目", self.num_items, "电影数目", self.num_users, "负样本数目", self.test_negative, "/n")
        # print("用户数目", self.num_items, "电影数目", self.num_users, "负样本数目", self.test_negative, "/n")


        assert len(self.test_ratings) == len(self.test_negative)
        self.train_dict = self.get_train_dict()


    
    def load_negative_file(self, filename):
        #return[ [int,int....],[int,int...]....]
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

# 训练的数据，一批有用户，有电影 有评分（有监督学习）
#测试的数据，一批有用户，有电影，无评分
    def load_test_rating_file_as_list(self, filename):
        #return [[user,item],[user,item]....]
        test_ratings = []
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                num_users = max(num_users, user)
                num_items = max(num_items, item)
                test_ratings.append([user, item])
                line = f.readline()
            test_num_users = num_users + 1
            test_num_items = num_items + 1
        return test_ratings, test_num_users, test_num_items

    def load_train_rating_file_as_list(self, filename):
        # return: [[user, item, rating],[user, item, rating].....]
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            max_items = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        train_num_users = num_users + 1
        train_num_items = num_items + 1
        # Construct matrix
        train_ratings = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                train_ratings.append([user, item, rating])
                line = f.readline()    
        return train_ratings, train_num_users, train_num_items
#得到下标的列表为crs构造矩阵提供方便
    def get_user_item_matrix_indices(self):
        #return [[user下标的列表],[item下标的列表],[全1的列表],]
        user_indices, item_indices, ratings = [], [], []
        for i in self.train_ratings:
            user_indices.append(i[0])
            item_indices.append(i[1])
            ratings.append(1)
        return [np.array(user_indices), np.array(item_indices), np.array(ratings)]


#一个用户多个电影多个分数；一个电影，多个用户多个分数对应关系
    def get_user_item_interact_list(self):
        #return[
        #     [   [user1,user1,user1]
        #       [item1,item2,item3]
        #       [rate1,rate2,rate3]
        #      ]
        #       .....
        #       ]
        user_item_interact = []
        user, item, rate = [], [], []
        user_idx = int(0)
        for i in self.train_ratings:
            print(i[0])
            if user_idx != i[0]:
                user_item_interact.append([user, item, rate])
                user_idx += 1
                user, item, rate = [], [], []
            else:
                user.append(i[0])
                item.append(i[1])
                rate.append(i[2])
        return user_item_interact

    def get_item_user_interact_list(self):
        # return[
        #     [
        #       [item1,item1,item1]
        #       [user1, user2,user3]
        #       [rate1,rate2,rate3]
        #      ]
        #       .....
        #       ]
        item_user_interact = []
        user, item, rate = [], [], []
        item_idx = 0
        for i in self.train_ratings:
            if item_idx != i[1]:
                item_user_interact.append([user, item, rate])
                item_idx += 1
                user, item, rate = [], [], []
            else:
                user.append(i[0])
                item.append(i[1])
                rate.append(i[2])
        return item_user_interact

    def get_train_instances(self, num_negative):
        #[  [user1,user1.......](一个带标签的数据，生成num_negative个负样本）
        #   [itemA,itemS.......](第一个是真实，剩下的全都是随机生成的）
        #   [ratex,0,0,.......]
        #]
        user, item, rate = [], [], []
        for i in self.train_ratings:
            user.append(i[0])
            item.append(i[1])
            rate.append(1)
            for t in range(num_negative):
                j = np.random.randint(self.num_items)
                while (i[0], j) in self.train_dict:
                    j = np.random.randint(self.num_items)
                user.append(i[0])
                item.append(j)
                rate.append(0)
        return [np.array(user), np.array(item), np.array(rate)]


    def get_user_and_item_matrix(self):
        #
        rom = np.random.rand(1, 100)
        user_matrix = self.user_item_matrix
        item_matrix = self.user_item_matrix.T
        return user_matrix, item_matrix

    def get_train_dict(self):
        #字典，键的类型不能是列表和字典，其他都可，这里的键是元组
        #{（用户1，电影A）：评分x，（用户2，电影B）：评分y，......}
        data_dict = {}
        for i in self.train_ratings:
            data_dict[(i[0], i[1])] = i[2]
        return data_dict

    def get_item_sparse_matrix(self):
        #有评分的 矩阵值都是1，没评分的都是0 电影和用户编号都是从小到大
        num_users, num_items = self.num_users, self.num_items
        user_indices, item_incides, rating_data = self.user_item_rating_indices
        #csr_matrix((所有非0数据的列表，（行下标，列下标）)，shape(行数，列数))
        item_sparse_matrix = csr_matrix((rating_data, (item_incides, user_indices)), shape=(num_items, num_users))
        return item_sparse_matrix

    def get_user_sparse_matrix(self):
        #是电影矩阵的转置
        user_sparse_matrix = self.get_item_sparse_matrix().T
        return user_sparse_matrix

#测试数据输出
# data=Dataset("data/ml-1m");
# print("用户数目", data.num_items, "电影数目", data.num_users)
# print("训练列表", data.train_ratings[:5], "\n")
# print("测试列表", data.test_ratings[:5], "\n")
# for i in range(4):
#     print("用户—电影下标对", data.user_indices[i], data.item_incides[i],data.rating_data[i])
#
# print("用户矩阵", data.user_matrix[1])
# print("电影矩阵", data.item_matrix[1])




