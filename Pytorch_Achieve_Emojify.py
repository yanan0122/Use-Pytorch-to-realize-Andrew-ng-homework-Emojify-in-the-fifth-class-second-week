import numpy as np
import torch

from emo_utils import *
from pytorch_model import *

np.random.seed(1)
np.random.seed(0)


def train(X_train, Y_train, word_to_index, word_to_vec_map, maxLen):
    net = emoji_model(word_to_vec_map, word_to_index)
    print(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    print(device)
    running_loss = 0.0
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)
    total_train_step = 0
    epoch = 3000
    # 当epoch=10000时，train_accuracy = 97.727273, test_accuracy = 85.714286。
    # 当epoch=5000时，train_accuracy = 99.242424, test_accuracy = 91.071429。
    # 当epoch=3000时，train_accuracy = 92.424242, test_accuracy = 78.571429。
    batch_size = 32
    for i in range(epoch):
        # for x, y in zip(X_train_indices, Y_train_oh):
        # if (i + 1)*batch_size>X_train_indices.shape[0]:
        #     x = X_train_indices[i * batch_size : -1, :]
        #     y = Y_train_oh[i * batch_size : -1, :]
        # else:
        x = X_train_indices[(i % 5) * batch_size:((i % 5) + 1) * batch_size, :]
        y = Y_train_oh[(i % 5) * batch_size:((i % 5) + 1) * batch_size, :]  # 有坑！
        x = torch.LongTensor(x)

        # y = np.reshape(y, (1, len(y)))
        y = torch.Tensor(y)
        x = x.to(device)
        y = y.to(device)
        outputs = net(x)
        # outputs = outputs.view(1, len(outputs))
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # total_train_step += 1
        # 训练每100个epoch打印
        if i % 500 == 0:
            print("-----------第{}轮训练结束-----------".format(i))
            print("训练次数：{}, Loss:{}".format(i, loss.item()))
    torch.save(net, "pytorch_emojify_3000.pth")


def accuracy(X_train, Y_train, X_test, Y_test, word_to_index, maxLen):
    """
    加载模型，返回模型在训练集和测试集上的精确度
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load("pytorch_emojify_3000.pth", map_location=device)
    model.eval()
    # model.to(device)

    # 设置训练集的输入,得到输出
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)  # 将X_train从一个一个单词转化成一个一个one-hot中序号
    X_train_indices = torch.LongTensor(X_train_indices)  # 将X_train_indices转成tensor
    X_train_indices = X_train_indices.to(device)  # 将X_train_indices移动到device上
    X_train_outputs = model(X_train_indices)  # 将所有输入通过网络，得到输出
    _, X_train_outputs = torch.max(X_train_outputs, dim=1)  # 选出每一个句子的输出中概率最大表情的序号
    Y_train = torch.Tensor(Y_train)
    Y_train = Y_train.to(device)

    # 设置测试集的输入,得到输出
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    X_test_indices = torch.LongTensor(X_test_indices)
    X_test_indices = X_test_indices.to(device)
    X_test_outputs = model(X_test_indices)
    _, X_test_outputs = torch.max(X_test_outputs, dim=1)
    Y_test = torch.Tensor(Y_test)
    Y_test = Y_test.to(device)

    train_acc, test_acc = 0, 0
    for i in range(len(X_train)):
        if X_train_outputs[i] == Y_train[i]:
            train_acc += 1
    train_acc = train_acc / len(X_train)

    for i in range(len(X_test)):
        if X_test_outputs[i] == Y_test[i]:
            test_acc += 1
    test_acc = test_acc / len(X_test)
    return train_acc, test_acc


def predict(word_to_index, maxLen):
    s = "I am so hungry"
    s2 = " "
    X = np.array([s, s2])  # 单独用一个s不能正常输出，非常非常奇怪
    # X.reshape((1, 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load("pytorch_emojify_5000.pth", map_location=device)
    model.eval()
    X = sentences_to_indices(X, word_to_index, maxLen)
    X = torch.LongTensor(X)
    X = X.to(device)
    Y = model(X)
    _, Y = torch.max(Y, dim=1)
    Y = Y.to("cpu")
    Y = Y.numpy()
    Y = Y[0]
    print(s + label_to_emoji(Y))


if __name__ == '__main__':
    # X_train, Y_train = read_csv('data/train_emoji.csv')
    # X_test, Y_test = read_csv('data/tesss.csv')
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    #
    # aa = max(X_train, key=len)  # 获取X_train中最长的句子
    # maxLen = len(aa.split())  # 获取句子的长度，split可以把string类型的句子按空格分开
    # maxLen = 10
    # train(X_train, Y_train, word_to_index, word_to_vec_map, maxLen)
    # train_acc, test_acc = accuracy(X_train, Y_train, X_test, Y_test, word_to_index, maxLen)
    # print("train_accuracy = %f, test_accuracy = %f" % (train_acc*100, test_acc*100))
    predict(word_to_index, 10)
