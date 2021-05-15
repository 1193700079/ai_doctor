import torch
import torch.nn as nn
from rnn_model import RNN

input_size = 768
hidden_size = 128
n_categories = 2

input = torch.rand(1, input_size)
hidden = torch.rand(1, hidden_size)

rnn = RNN(input_size, hidden_size, n_categories)
outputs, hidden = rnn(input, hidden)
# print("outputs:", outputs)
# print("hidden:", hidden)

import time
import math

def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

# 假定模型训练开始时间是10min之前
since = time.time() - 10*60
period = timeSince(since)
print(period)
