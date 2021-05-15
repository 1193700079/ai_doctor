# 导入包
import json
import numpy as np
# 创建训练数据集, 从原始训练文件中将中文字符进行数字编码, 并将标签页进行数字编码
def create_train_data(train_data_file, result_file, json_file, tag2id, max_length=20):
    # 导入json格式的中文字符到id的映射表
    char2id = json.load(open(json_file, mode='r', encoding='utf-8'))
    char_data, tag_data = [], []
    # 打开原始训练文件
    with open(train_data_file, mode='r', encoding='utf-8') as f:
        # 初始化一条语句数字化编码后的列表
        char_ids = [0] * max_length
        tag_ids = [0] * max_length
        idx = 0
        for line in f.readlines():
            line = line.strip('\n').strip()
            # 如果不是空行, 并且当前语句长度没有超过max_length, 则进行字符到id的映射
            if len(line) > 0 and line and idx < max_length:
                ch, tag = line.split('\t')
                # 如果当前字符存在于映射表中, 则直接映射为对应的id值
                if char2id.get(ch):
                    char_ids[idx] = char2id[ch]
                # 否则直接用"UNK"的id值来代替这个未知字符
                else:
                    char_ids[idx] = char2id['UNK']
                # 将标签也进行对应的转换
                tag_ids[idx] = tag2id[tag]
                idx += 1
            # 如果是空行, 或者当前语句长度超过max_length
            else:
                # 如果当前语句长度超过max_length, 直接将[0: max_langth]的部分作为结果
                if idx <= max_length:
                    char_data.append(char_ids)
                    tag_data.append(tag_ids)
                # 遇到空行, 说明当前句子已经结束, 初始化清零, 为下一个句子的映射做准备
                char_ids = [0] * max_length
                tag_ids = [0] * max_length
                idx = 0
    # 将数字化编码后的数据封装成numpy的数组类型, 数字编码采用np.int32
    x_data = np.array(char_data, dtype=np.int32)
    y_data = np.array(tag_data, dtype=np.int32)
    # 直接利用np.savez()将数据存储为.npz类型的文件
    np.savez(result_file, x_data=x_data, y_data=y_data)
    print("create_train_data Finished!".center(100, "-"))
# 参数1:字符码表文件路
json_file = 'data/char_to_id.json'
# 参数2:标签码表对照字典
tag2id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}
# 参数3:训练数据文件路径
train_data_file = 'data/back_train.txt'
# 参数4:创建的npz文件保路径(训练数据)
result_file = 'data/train.npz'
if __name__ == '__main__':
    create_train_data(train_data_file, result_file, json_file, tag2id)




