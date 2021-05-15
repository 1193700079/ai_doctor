import os
import torch
import torch.nn as nn

# 导入RNN模型结构
from rnn_model import RNN
# 导入bert预训练模型编码函数
from bert_chinese_encode import get_bert_encode_for_single

# 预加载的模型参数路径
MODEL_PATH = 'BERT_RNN.pth'

# 隐层节点数, 输入层尺寸, 类别数都和训练时相同即可
n_hidden = 128
input_size = 768
n_categories = 2

# 实例化RNN模型, 并加载保存模型参数
rnn = RNN(input_size, n_hidden, n_categories)
rnn.load_state_dict(torch.load(MODEL_PATH))


def _test(line_tensor):
    """模型测试函数, 它将用在模型预测函数中, 用于调用RNN模型并返回结果.它的参数line_tensor代表输入文本的张量表示"""
    # 初始化隐层张量
    hidden = rnn.initHidden()
    # 与训练时相同, 遍历输入文本的每一个字符
    for i in range(line_tensor.size()[0]):
        # 将其逐次输送给rnn模型
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    # 获得rnn模型最终的输出
    return output


def predict(input_line):
    """模型预测函数, 输入参数input_line代表需要预测的文本"""
    # 不自动求解梯度
    with torch.no_grad():
        # 将input_line使用bert模型进行编码
        output = _test(get_bert_encode_for_single(input_line))
        # 从output中取出最大值对应的索引, 比较的维度是1
        _, topi = output.topk(1, 1)
        # 返回结果数值
        return topi.item()

input_line = "点瘀样尖针性发多"
result = predict(input_line)
print("result:", result)


# def batch_predict(input_path, output_path):
#     """批量预测函数, 以原始文本(待识别的命名实体组成的文件)输入路径
#        和预测过滤后(去除掉非命名实体的文件)的输出路径为参数"""
#     # 待识别的命名实体组成的文件是以疾病名称为csv文件名,
#     # 文件中的每一行是该疾病对应的症状命名实体
#     # 读取路径下的每一个csv文件名, 装入csv列表之中
#     csv_list = os.listdir(input_path)
#     # 遍历每一个csv文件
#     for csv in csv_list:
#         # 以读的方式打开每一个csv文件
#         with open(os.path.join(input_path, csv), "r" ,encoding="utf-8") as fr:
#             # 再以写的方式打开输出路径的同名csv文件
#             with open(os.path.join(output_path, csv), "w") as fw:
#                 # 读取csv文件的每一行
#                 input_line = fr.readline()
#                 # 使用模型进行预测
#                 res = predict(input_line)
#                 # 如果结果为1
#                 if res:
#                     # 说明审核成功, 写入到输出csv中
#                     print("ok")
#                     fw.write(input_line + "\n")
#                 else:
#                     pass
#
# input_path = "structured/noreview"
# output_path = "structured/reviewed"
# batch_predict(input_path, output_path)
