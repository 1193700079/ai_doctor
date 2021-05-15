# -*- coding:UTF-8 -*-
import time
import pandas as pd
from sklearn.utils import shuffle
from functools import reduce
from collections import Counter
from bert_chinese_encode import get_bert_encode
import torch
import torch.nn as nn
# 定义数据加载器构造函数
def data_loader(data_path, batch_size, split=0.2):
    '''
    data_path: 训练数据的路径
    batch_size: 训练集和验证集的批次大小
    split: 训练集和验证集的划分比例
    return: 训练数据生成器, 验证数据的生成器, 训练数据的大小, 验证数据的大小
    '''
    # 首先读取数据
    data = pd.read_csv(data_path, header=None, sep="\t")
    # 打印一下整体数据集上正负样本的数量
    print("数据集的正负样本数量:")
    print(dict(Counter(data[0].values)))
    # 要对读取的数据进行散乱顺序的操作
    data = shuffle(data).reset_index(drop=True)
    # 划分训练集和验证集
    split_point = int(len(data) * split)
    valid_data = data[:split_point]
    train_data = data[split_point:]
    # 保证验证集中的数据总数至少能够满足一个批次
    if len(valid_data) < batch_size:
        raise("Batch size or split not match!")
    # 定义获取每个批次数据生成器的函数
    def _loader_generator(data):
        # data: 训练数据或者验证数据
        # 以每个批次大小的间隔来遍历数据集
        for batch in range(0, len(data), batch_size):
            # 初始化batch数据的存放张量列表
            batch_encoded = []
            batch_labels = []
            # 逐条进行数据的遍历
            for item in data[batch: batch + batch_size].values.tolist():
                # 对每条数据进行bert预训练模型的编码
                encoded = get_bert_encode(item[1], item[2])
                # 将编码后的每条数据放进结果列表中
                batch_encoded.append(encoded)
                # 将标签放入结果列表中
                batch_labels.append([item[0]])
            # 使用reduce高阶函数将列表中的数据转换成模型需要的张量形式
            # encoded的形状[batch_size, 2 * max_len, embedding_size]
            encoded = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_encoded)
            labels = torch.tensor(reduce(lambda x, y: x + y, batch_labels))
            # 以生成器的方式返回数据和标签
            yield (encoded, labels)

    return _loader_generator(train_data), _loader_generator(valid_data), len(train_data), len(valid_data)


data_path = "./train_data.csv"
batch_size = 32
max_len = 10

train_data_labels, valid_data_labels, train_data_length, valid_data_length = data_loader(data_path, batch_size)
# print(next(train_data_labels))
# print(next(valid_data_labels))
# print("train_data_length:", train_data_length)
# print("valid_data_length:", valid_data_length)


from finetuning_net import Net
import torch.optim as optim
# 初始化若干参数
embedding_size = 768
char_size = 2 * max_len
# 实例化微调网络
net = Net(embedding_size, char_size)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def train(train_data_labels):
    # train_data_labels: 代表训练数据和标签的生成器对象
    # return: 整个训练过程的平均损失和, 正确标签数量的累加和
    # 初始化损失变量和准确数量
    train_running_loss = 0.0
    train_running_acc = 0.0
    # 遍历数据生成器
    for train_tensor, train_labels in train_data_labels:
        # 首先将优化器的梯度归零
        optimizer.zero_grad()
        # 将训练数据传入模型得到输出结果
        train_outputs = net(train_tensor)
        # 计算当前批次的平均损失
        train_loss = criterion(train_outputs, train_labels)
        # 累加损失
        train_running_loss += train_loss.item()
        # 训练模型, 反向传播
        train_loss.backward()
        # 优化器更新模型参数
        optimizer.step()
        # 将该批次样本中正确的预测数量进行累加
        train_running_acc += (train_outputs.argmax(1) == train_labels).sum().item()

    # 整个循环结束后, 训练完毕, 得到损失和以及正确样本的总量
    return train_running_loss, train_running_acc


def valid(valid_data_labels):
    # valid_data_labels: 代表验证数据和标签的生成器对象
    # return: 整个验证过程中的平均损失和和正确标签的数量和
    # 初始化损失值和正确标签数量
    valid_running_loss = 0.0
    valid_running_acc = 0

    # 循环遍历验证数据集的生成器
    for valid_tensor, valid_labels in valid_data_labels:
        # 测试阶段梯度不被更新
        with torch.no_grad():
            # 将特征输入网络得到预测张量
            valid_outputs = net(valid_tensor)
            # 计算当前批次的损失值
            valid_loss = criterion(valid_outputs, valid_labels)
            # 累加损失和
            valid_running_loss += valid_loss.item()
            # 累加正确预测的标签数量
            valid_running_acc += (valid_outputs.argmax(1) == valid_labels).sum().item()

    # 返回整个验证过程中的平均损失和, 累加的正确标签数量
    return valid_running_loss, valid_running_acc
epochs = 20
# 定义每个轮次的损失和准确率的列表初始化, 用于未来画图
all_train_losses = []
all_valid_losses = []
all_train_acc = []
all_valid_acc = []
for epoch in range(epochs):
    # 打印轮次
    print("Epoch:", epoch + 1)
    # 首先通过数据加载函数, 获得训练数据和验证数据的生成器, 以及对应的训练样本数和验证样本数
    train_data_labels, valid_data_labels, train_data_len, \
    valid_data_len = data_loader(data_path, batch_size)

    # 调用训练函数进行训练
    train_running_loss, train_running_acc = train(train_data_labels)
    # 调用验证函数进行验证
    valid_running_loss, valid_running_acc = valid(valid_data_labels)

    # 计算平均损失, 每个批次的平均损失之和乘以批次样本数量, 再除以本轮次的样本总数
    train_average_loss = train_running_loss * batch_size / train_data_len
    valid_average_loss = valid_running_loss * batch_size / valid_data_len

    # 计算准确率, 本轮次总的准确样本数除以本轮次的总样本数
    train_average_acc = train_running_acc / train_data_len
    valid_average_acc = valid_running_acc / valid_data_len

    # 接下来将4个值添加进画图的列表中
    all_train_losses.append(train_average_loss)
    all_valid_losses.append(valid_average_loss)
    all_train_acc.append(train_average_acc)
    all_valid_acc.append(valid_average_acc)

    # 打印本轮次的训练损失, 准确率, 以及验证损失, 准确率
    print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
    print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)

print("Finished Training.")
 

# 导入画图的工具包
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 创建第一张画布
plt.figure(0)

# 绘制训练损失曲线   颜色默认为蓝色
plt.plot(all_train_losses, label="Train Loss")
# 绘制验证损失曲线, 同时将颜色设置为红色
plt.plot(all_valid_losses, color="red", label="Valid Loss")

# 定义横坐标间隔对象, 间隔等于1, 代表一个轮次一个坐标点
x_major_locator = MultipleLocator(1)
# 获得当前坐标图的句柄
ax = plt.gca()
# 在句柄上设置横坐标的刻度间隔
ax.xaxis.set_major_locator(x_major_locator)
# 设置横坐标取值范围
plt.xlim(1, epochs)
# 将图例放在左上角
plt.legend(loc='upper left')
# 保存图片
plt.savefig("./loss.png")


# 创建第二张画布
plt.figure(1)

# 绘制训练准确率曲线
plt.plot(all_train_acc, label="Train Acc")
# 绘制验证准确率曲线, 同时将颜色设置为红色
plt.plot(all_valid_acc, color="red", label="Valid Acc")
# 定义横坐标间隔对象, 间隔等于1, 代表一个轮次一个坐标点
x_major_locator = MultipleLocator(1)
# 获得当前坐标图的句柄
ax = plt.gca()
# 在句柄上设置横坐标的刻度间隔
ax.xaxis.set_major_locator(x_major_locator)
# 设置横坐标的取值范围
plt.xlim(1, epochs)
# 将图例放在左上角
plt.legend(loc='upper left')
# 保存图片
plt.savefig("./acc.png")


# 保存模型时间
time_ = int(time.time())
# 设置保存路径和模型名称
MODEL_PATH = './YRQ_BERT_net_%d.pth' % time_
# 保存模型
torch.save(net.state_dict(), MODEL_PATH)

















































