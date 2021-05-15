# -*- coding:UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
# 创建微调模型类Net
class Net(nn.Module):
    def __init__(self, char_size=20, embedding_size=768, dropout=0.2):
        '''
        char_size: 输入句子中的字符数量, 因为在bert继承中规范化后的句子长度为10, 所以这里面等于两个句子的长度2*char_size
        embedding_size: 字嵌入的维度, 因为使用了bert中文模型, 而bert的嵌入维度是768, 因此这里的词嵌入维度也要是768
        dropout: 为了防止过拟合, 网络中引入dropout层, dropout为置0的比例, 默认等于0.2
        '''
        super(Net, self).__init__()
        # 将传入的参数变成类内部的变量
        self.char_size = char_size
        self.embedding_size = embedding_size
        # 实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 定义第一个全连接层
        self.fc1 = nn.Linear(char_size * embedding_size, 8)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        # 首先要对输入的张量形状进行变化, 要满足匹配全连接层
        x = x.view(-1, self.char_size * self.embedding_size)
        # 使用Dropout层
        x = self.dropout(x)
        # 将x输入进第一个全连接层
        x = F.relu(self.fc1(x))
        # 再次使用Dropout层
        x = self.dropout(x)
        # 将x输入进第二个全连接层
        x = F.relu(self.fc2(x))
        return x


# embedding_size = 768
# char_size = 20
# dropout = 0.2
#
# x = torch.randn(1, 20, 768)
#
# net = Net(char_size, embedding_size, dropout)
# res = net(x)
# print(res)



