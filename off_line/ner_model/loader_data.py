# 导入相关的包
import numpy as np
import torch
import torch.utils.data as Data
# 生成批量训练数据
def load_dataset(data_file, batch_size):
    # 将第二步生成的train.npz文件导入内存
    data = np.load(data_file)
    # 分别取出特征值和标签
    x_data = data['x_data']
    y_data = data['y_data']
    # 将数据封装成tensor张量
    x = torch.tensor(x_data, dtype=torch.long)
    y = torch.tensor(y_data, dtype=torch.long)
    # 将数据封装成Tensor数据集
    dataset = Data.TensorDataset(x, y)
    total_length = len(dataset)
    # 采用80%的数据作为训练集, 20%的数据作为测试集
    train_length = int(total_length * 0.8)
    validation_length = total_length - train_length
    # 利用Data.random_split()直接切分集合, 按照80%, 20%的比例划分
    train_dataset, validation_dataset = Data.random_split(dataset=dataset,
                                        lengths=[train_length, validation_length])
    # 将训练集进行DataLoader封装
    # 参数说明如下:
    # dataset:     训练数据集
    # batch_size:  代表批次大小, 若数据集总样本数量无法被batch_size整除, 则最后一批数据为余数
    #              若设置drop_last为True， 则自动抹去最后不能被整除的剩余批次
    # shuffle:     是否每个批次为随机抽取, 若为True, 则每次迭代时数据为随机抽取
    # num_workers: 设定有多少子进程用来做数据加载, 默认为0, 即数据将被加载到主进程中
    # drop_last:   是否去除不能被整除后的最后批次, 若为True, 则不生成最后不能被整除剩余的数据内容
    #              例如: dataset长度为1028, batch_size为8,
    #              若drop_last=True, 则最后剩余的4(1028/8=128余4)条数据将被抛弃不用
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4, drop_last=True)
    validation_loader = Data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=4, drop_last=True)
    # 将两个数据生成器封装为一个字典类型
    data_loaders = {'train': train_loader, 'validation': validation_loader}
    # 将两个数据集的长度也封装为一个字典类型
    data_size = {'train': train_length, 'validation': validation_length}
    return data_loaders, data_size
# 批次大小
BATCH_SIZE = 8
# 编码后的训练数据文件路径
DATA_FILE = 'data/train.npz'
if __name__ == '__main__':
    data_loader, data_size = load_dataset(DATA_FILE, BATCH_SIZE)
    print('data_loader:', data_loader, '\ndata_size:', data_size)