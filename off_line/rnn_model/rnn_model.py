import torch
import torch.nn as  nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """初始化函数中有三个参数,分别是
        输入张量最后一维的尺寸大小,
        隐层张量最后一维的尺寸大小,
        输出张量最后一维的尺寸大小    """
        super(RNN, self).__init__()
        # 传入隐含层尺寸大小
        self.hidden_size = hidden_size
        # 构建从输入到隐含层的线性变化, 这个线性层的输入尺寸是input_size + hidden_size
        # 这是因为在循环网络中, 每次输入都有两部分组成，分别是此时刻的输入和上一时刻产生的输出.
        # 这个线性层的输出尺寸是hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 构建从输入到输出层的线性变化, 这个线性层的输入尺寸还是input_size + hidden_size
        # 这个线性层的输出尺寸是output_size.
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # 最后需要对输出做softmax处理, 获得结果. 归一化
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """在forward函数中, 参数分别是规定尺寸的输入张量, 以及规定尺寸的初始化隐层张量"""
        # 首先使用torch.cat将input与hidden进行张量拼接  1在列方向上进行拼接
        combined = torch.cat((input, hidden), 1)
        # 通过输入层到隐层变换获得hidden张量
        hidden = self.i2h(combined)
        # 通过输入到输出层变换获得output张量
        output = self.i2o(combined)
        # 对输出进行softmax处理
        output = self.softmax(output)
        # 返回输出张量和最后的隐层结果
        return output, hidden

    def initHidden(self):
        """隐层初始化函数"""
        # 将隐层初始化成为一个1xhidden_size的全0张量
        return torch.zeros(1, self.hidden_size)

# 选取损失函数为NLLLoss()  等价于 交叉熵
criterion = nn.NLLLoss()
# 学习率为0.005
learning_rate = 0.005






