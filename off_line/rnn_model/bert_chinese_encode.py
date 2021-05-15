import torch
import torch.nn as nn

# 通过torch.hub(pytorch中专注于迁移学的工具)获得已经训练好的bert-base-chinese模型
model =  torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinese')

# 获得对应的字符映射器, 它将把中文的每个字映射成一个数字
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')

def get_bert_encode_for_single(text):
    """
    description: 使用bert-chinese编码中文文本
    :param text: 要进行编码的文本
    :return: 使用bert编码后的文本张量表示
    """
    # 首先使用字符映射器对每个汉字进行映射
    # 这里需要注意, bert的tokenizer映射后会为结果前后添加开始和结束标记即101和102
    # 这对于多段文本的编码是有意义的, 但在我们这里没有意义, 因此使用[1:-1]对头和尾进行切片
    indexed_tokens = tokenizer.encode(text)[1:-1]
    # 之后将列表结构转化为tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    # 使模型不自动计算梯度
    with torch.no_grad():
        # 调用模型获得隐层输出
        encoded_layers = model(tokens_tensor)
    # 输出的隐层是一个三维张量, 最外层一维是1,  我们使用[0]降去它.
    encoded_layers = encoded_layers[0]
    return encoded_layers[0]
if __name__ == '__main__':
    text = "你好, 华北理工大学"
    outputs = get_bert_encode_for_single(text)
    print(outputs)
    print(outputs.shape)
