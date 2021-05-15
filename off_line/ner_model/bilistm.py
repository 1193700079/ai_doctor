# 本段代码构建类BiLSTM, 完成初始化和网络结构的搭建
# 总共3层: 词嵌入层, 双向LSTM层, 全连接线性层
import torch
import torch.nn as nn

# 参数0:句子集合
sentence_list = [
    "确诊弥漫大b细胞淋巴瘤1年",
    "反复咳嗽、咳痰40年,再发伴气促5天。",
    "生长发育迟缓9年。",
    "右侧小细胞肺癌第三次化疗入院",
    "反复气促、心悸10年,加重伴胸痛3天。",
    "反复胸闷、心悸、气促2多月,加重3天",
    "咳嗽、胸闷1月余, 加重1周",
    "右上肢无力3年, 加重伴肌肉萎缩半年"]
# 参数1:码表与id对照
char_to_id = {"<PAD>":0}
# 参数2:标签码表对照
tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
# 参数3:字向量维度
EMBEDDING_DIM = 200
# 参数4:隐层维度
HIDDEN_DIM = 100
# 参数5:批次大小
BATCH_SIZE = 8
# 参数6:句子长度
SENTENCE_LENGTH = 20
# 参数7:堆叠 LSTM 层数
NUM_LAYERS = 1

# 本函数实现将中文文本映射为数字化的张量
def sentence_map(sentence_list, char_to_id, max_length):
    """
    description: 将句子中的每一个字符映射到码表中
    :param sentence: 待映射句子, 类型为字符串或列表
    :param char_to_id: 码表, 类型为字典, 格式为{"字1": 1, "字2": 2}
    :return: 每一个字对应的编码, 类型为tensor
    """
    # 字符串按照逆序进行排序, 不是必须操作
    sentence_list.sort(key=lambda c:len(c), reverse=True)
    # 定义句子映射列表
    sentence_map_list = []
    for sentence in sentence_list:
        # 生成句子中每个字对应的 id 列表
        sentence_id_list = [char_to_id[c] for c in sentence]
        # 计算所要填充 0 的长度
        padding_list = [0] * (max_length-len(sentence))
        # 组合
        sentence_id_list.extend(padding_list)
        # 将填充后的列表加入句子映射总表中
        sentence_map_list.append(sentence_id_list)
    # 返回句子映射集合, 转为标量
    return torch.tensor(sentence_map_list, dtype=torch.long)

class BiLSTM(nn.Module):
    """
    description: BiLSTM 模型定义
    """
    def __init__(self, vocab_size, tag_to_id, input_feature_size, hidden_size,
                 batch_size, sentence_length, num_layers=1, batch_first=True):
        """
        description: 模型初始化
        :param vocab_size:          所有文本的字符数量
        :param tag_to_id:           标签与 id 对照表
        :param input_feature_size:  输入的维度，字嵌入维度( 即LSTM输入层维度 input_size )
        :param hidden_size:         隐藏层向量维度
        :param batch_size:          批训练大小
        :param sentence_length:      句子长度的限制
        :param num_layers:           LSTM 层数大小
        :param batch_first:         是否将batch_size放置到矩阵的第一维度
        """
        # 类继承初始化函数
        super(BiLSTM, self).__init__()
        # 设置标签与id对照
        self.tag_to_id = tag_to_id
        # 设置标签大小, 对应BiLSTM最终输出分数矩阵宽度
        self.tag_size = len(tag_to_id)
        # 设定LSTM输入特征大小, 对应词嵌入的维度大小
        self.embedding_size = input_feature_size
        # 设置隐藏层维度, 若为双向时想要得到同样大小的向量, 需要除以2 ！！ 双向网络
        self.hidden_size = hidden_size // 2
        # 设置批次大小, 对应每个批次的样本条数, 可以理解为输入张量的第一个维度
        self.batch_size = batch_size
        # 设定句子长度
        self.sentence_length = sentence_length
        # 设定是否将batch_size放置到矩阵的第一维度, 取值True, 或False
        self.batch_first = batch_first
        # 设置网络的LSTM层数
        self.num_layers = num_layers

        # 网络模型
        # 构建词嵌入层: 字向量, 维度为总单词数量与词嵌入维度
        # 参数: 总体字库的单词数量, 每个字被嵌入的维度
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        # 构建双向LSTM层: BiLSTM (参数: input_size      字向量维度(即输入层大小),
        #                               hidden_size     隐藏层维度,
        #                               num_layers      层数,
        #                               bidirectional   是否为双向,
        #                               batch_first     是否批次大小在第一维)
        self.bilstm = nn.LSTM(input_size=input_feature_size,
                              hidden_size=self.hidden_size,
                              num_layers=num_layers,
                              bidirectional=True,
                              batch_first=batch_first)

        # 构建全连接线性层: 将BiLSTM的输出层进行线性变换
        self.linear = nn.Linear(hidden_size, self.tag_size)

    # 本函数实现类BiLSTM中的前向计算函数forward()
    def forward(self, sentences_sequence):
        """
        description: 将句子利用BiLSTM进行特征计算，分别经过Embedding->BiLSTM->Linear，
                     获得发射矩阵（emission scores）
        :param sentences_sequence: 句子序列对应的编码，
                                   若设定 batch_first 为 True，
                                   则批量输入的 sequence 的 shape 为(batch_size, sequence_length)
        :return:    返回当前句子特征，转化为 tag_size 的维度的特征
        """
        # 初始化隐藏状态值
        h0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        # 初始化单元状态值
        c0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        # 生成字向量， shape 为(batch, sequence_length, input_feature_size)
        # 注：embedding cuda 优化仅支持 SGD 、 SparseAdam
        input_features = self.embedding(sentences_sequence)

        # 将字向量与初始值(隐藏状态 h0 , 单元状态 c0 )传入 LSTM 结构中
        # 输出包含如下内容：
        # 1, 计算的输出特征，shape 为(batch, sentence_length, hidden_size)
        #    顺序为设定 batch_first 为 True 情况, 若未设定则 batch 在第二位
        # 2, 最后得到的隐藏状态 hn ， shape 为(num_layers * num_directions, batch, hidden_size)
        # 3, 最后得到的单元状态 cn ， shape 为(num_layers * num_directions, batch, hidden_size)
        output, (hn, cn) = self.bilstm(input_features, (h0, c0))
        # 将输出特征进行线性变换，转为 shape 为 (batch, sequence_length, tag_size) 大小的特征
        sequence_features = self.linear(output)
        # 输出线性变换为 tag 映射长度的特征
        return sequence_features

if __name__ == '__main__':
    for sentence in sentence_list:
        for _char in sentence:
            if _char not in char_to_id:
                char_to_id[_char] = len(char_to_id)
    sentence_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)

    model = BiLSTM(vocab_size=len(char_to_id), tag_to_id=tag_to_id, input_feature_size=EMBEDDING_DIM,
                   hidden_size=HIDDEN_DIM, batch_size=BATCH_SIZE, sentence_length=SENTENCE_LENGTH, num_layers=NUM_LAYERS)

    # 模型的参数就是forward的参数
    sentence_features = model(sentence_sequence)
    print("sequence_features:\n", sentence_features)
