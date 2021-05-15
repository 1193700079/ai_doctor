# -*- coding:UTF-8 -*-
# 导入若干工具包
from flask import Flask
from flask import request
app = Flask(__name__)

import torch
# 导入中文预训练模型的编码函数
from bert_chinese_encode import get_bert_encode
# 导入微调网络模型
from finetuning_net import Net

# 设置训练好的模型路径
MODEL_PATH = "/root/ai_doctor/on_line/bert_server/model/YRQ_BERT_net_1618668193.pth"

# 定义实例化的模型参数
embedding_size = 768
char_size = 20
dropout = 0.2

# 初始化微调模型
net = Net(embedding_size, char_size, dropout)
# 加载已经训练好的模型
net.load_state_dict(torch.load(MODEL_PATH))
# 因为是在线部分, 所以采用评估模式, 本质是模型参数不发生变化
net.eval()

# 定义服务请求的路径和方式
@app.route('/v1/recognition/', methods=["POST"])
def recognition():
    print("已经接受到传来的数据")
    # 首先接收数据
    text_1 = request.form['text1']
    text_2 = request.form['text2']
    # 对原始文本进行编码
    inputs = get_bert_encode(text_1, text_2, mark=102, max_len=10)
    # 使用微调模型进行预测
    outputs = net(inputs)
    # 从预测张量中获取结果
    _, predicted = torch.max(outputs, 1)
    print("两个文本的相关程度是 "+str(predicted.item()))
    # 返回字符串类型的结果
    return str(predicted.item())


