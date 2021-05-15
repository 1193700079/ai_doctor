# 导入相关包与模块
import torch
import torch.nn as nn

# 初始化的示例语句, 共8行, 可以理解为当前批次batch_size=8
from torch import optim

sentence_list = [
    "确诊弥漫大b细胞淋巴瘤1年",
    "反复咳嗽、咳痰40年,再发伴气促5天。",
    "生长发育迟缓9年。",
    "右侧小细胞肺癌第三次化疗入院",
    "反复气促、心悸10年,加重伴胸痛3天。",
    "反复胸闷、心悸、气促2多月,加重3天",
    "咳嗽、胸闷1月余, 加重1周",
    "右上肢无力3年, 加重伴肌肉萎缩半年"
]

# 开始字符和结束字符
START_TAG = "<START>"
STOP_TAG = "<STOP>"
# 标签和序号的对应码表
tag_to_ix = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}
# 词嵌入的维度
EMBEDDING_DIM = 200
# 隐藏层神经元的数量
HIDDEN_DIM = 100
# 批次的大小
BATCH_SIZE = 8
# 设置最大语句限制长度
SENTENCE_LENGTH = 20
# 默认神经网络的层数
NUM_LAYERS = 1
# 初始化的字符和序号的对应码表
char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
              "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                       num_layers, batch_size, sequence_length):
        '''
        description: 模型初始化
        :param vocab_size:          所有句子包含字符大小
        :param tag_to_ix:           标签与id对照字典
        :param embedding_dim:       字嵌入维度(即LSTM输入层维度input_size)
        :param hidden_dim:          隐藏层向量维度
        :param num_layers:          神经网络的层数
        :param batch_size:          批次的数量
        :param sequence_length:     语句的限制最大长度
        '''
        # 继承函数的初始化
        super(BiLSTM_CRF, self).__init__()
        # 设置标签与id对照
        self.tag_to_ix = tag_to_ix
        # 设置标签大小，对应 BiLSTM 最终输出分数矩阵宽度
        self.tagset_size = len(tag_to_ix)
        # 设定 LSTM 输入特征大小
        self.embedding_dim = embedding_dim
        # 设置隐藏层维度
        self.hidden_dim = hidden_dim
        # 设置单词总数的大小
        self.vocab_size = vocab_size
        # 设置隐藏层的数量
        self.num_layers = num_layers
        # 设置语句的最大限制长度
        self.sequence_length = sequence_length
        # 设置批次的大小
        self.batch_size = batch_size

        # 构建词嵌入层, 两个参数分别是单词总数, 词嵌入维度
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # 构建双向LSTM层, 输入参数包括词嵌入维度, 隐藏层大小, 堆叠的LSTM层数, 是否双向标志位
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=self.num_layers, bidirectional=True)

        # 构建全连接线性层, 一端对接LSTM隐藏层, 另一端对接输出层, 相应的维度就是标签数量tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 初始化转移矩阵, 转移矩阵是一个方阵[tagset_size, tagset_size]
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 按照损失函数小节的定义, 任意的合法句子不会转移到"START_TAG", 因此设置为-10000
        # 同理, 任意合法的句子不会从"STOP_TAG"继续向下转移, 也设置为-10000
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # 初始化隐藏层, 利用单独的类函数init_hidden()来完成
        self.hidden = self.init_hidden()

    # 定义类内部专门用于初始化隐藏层的函数
    def init_hidden(self):
        # 为了符合LSTM的输入要求, 我们返回h0, c0, 这两个张量的shape完全一致
        # 需要注意的是shape: [2 * num_layers, batch_size, hidden_dim / 2]
        return (torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2),
                 torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2))

    # 在类中将文本信息经过词嵌入层, BiLSTM层, 线性层的处理, 最终输出句子张量
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # a = self.word_embeds(sentence)
        # print(a.shape)  torch.Size([8, 20, 200])
        # LSTM的输入要求形状为 [sequence_length, batch_size, embedding_dim]
        # LSTM的隐藏层h0要求形状为 [num_layers * direction, batch_size, hidden_dim]
        embeds = self.word_embeds(sentence).view(self.sequence_length, self.batch_size, -1)

        # LSTM的两个输入参数: 词嵌入后的张量, 随机初始化的隐藏层张量
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # 要保证输出张量的shape: [sequence_length, batch_size, hidden_dim]
        lstm_out = lstm_out.view(self.sequence_length, self.batch_size, self.hidden_dim)

        # 将BiLSTM的输出经过一个全连接层, 得到输出张量shape:[sequence_length, batch_size, tagset_size]
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 计算损失函数第一项的分值函数, 本质上是发射矩阵和转移矩阵的累加和
    def _forward_alg(self, feats):
        # 初始化一个alphas张量, 代表转移矩阵的起始位置
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # init_alphas: [1, 7] , [-10000, -10000, -10000, -10000, -10000, -10000, -10000]
        # 仅仅把START_TAG赋值为0, 代表着接下来的转移只能从START_TAG开始
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # 前向计算变量的赋值, 这样在反向求导的过程中就可以自动更新参数
        forward_var = init_alphas
        # 输入进来的feats: [20, 8, 7], 为了接下来按句子进行计算, 要将batch_size放在第一个维度上
        feats = feats.transpose(1, 0)
        # feats: [8, 20, 7]是一个3维矩阵, 最外层代表8个句子, 内层代表每个句子有20个字符,
        # 每一个字符映射成7个标签的发射概率
        # 初始化最终的结果张量, 每个句子对应一个分数
        result = torch.zeros((1, self.batch_size))
        idx = 0
        # 按行遍历, 总共循环batch_size次
        for feat_line in feats:
            # 遍历一行语句, 每一个feat代表一个time_step
            for feat in feat_line:
                # 当前time_step的一个forward tensors
                alphas_t = []
                # 在当前time_step, 遍历所有可能的转移标签, 进行累加计算
                for next_tag in range(self.tagset_size):
                    # 广播发射矩阵的分数
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                    # 第i个time_step循环时, 转移到next_tag标签的转移概率
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # 将前向矩阵, 转移矩阵, 发射矩阵累加
                    next_tag_var = forward_var + trans_score + emit_score
                    # 计算log_sum_exp()函数值
                    # a = log_sum_exp(next_tag_var), 注意: log_sum_exp()函数仅仅返回一个实数值
                    # print(a.shape) : tensor(1.0975) , ([])
                    # b = a.view(1) : tensor([1.0975]), 注意: a.view(1)的操作就是将一个数字变成一个一阶矩阵, ([]) -> ([1])
                    # print(b.shape) : ([1])
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                # print(alphas_t) : [tensor([337.6004], grad_fn=<ViewBackward>), tensor([337.0469], grad_fn=<ViewBackward>), tensor([337.8497], grad_fn=<ViewBackward>), tensor([337.8668], grad_fn=<ViewBackward>), tensor([338.0186], grad_fn=<ViewBackward>), tensor([-9662.2734], grad_fn=<ViewBackward>), tensor([337.8692], grad_fn=<ViewBackward>)]
                # temp = torch.cat(alphas_t)
                # print(temp) : tensor([[  337.6004,   337.0469,   337.8497,   337.8668,   338.0186, -9662.2734, 337.8692]])
                # 将列表张量转变为二维张量
                forward_var = torch.cat(alphas_t).view(1, -1)
                # print(forward_var.shape) : [1, 7]
            # print(forward_var) : tensor([[   13.7928,    16.0067,    14.1092, -9984.7852,    15.8380]])
            # 添加最后一步转移到"STOP_TAG"的分数, 就完成了整条语句的分数计算
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            # print(terminal_var) : tensor([[  339.2167,   340.8612,   340.2773,   339.0194,   340.8908, -9659.5732, -9660.0527]])
            # 计算log_sum_exp()函数值, 作为一条样本语句的最终得分
            alpha = log_sum_exp(terminal_var)
            # print(alpha) : tensor(341.9394)
            # 将得分添加进结果列表中, 作为函数结果返回
            result[0][idx] = alpha
            idx += 1
        return result

    def _score_sentence(self, feats, tags):
        # feats: [20, 8, 7] , tags: [8, 20]
        # 初始化一个0值的tensor, 为后续累加做准备
        score = torch.zeros(1)
        # 将START_TAG和真实标签tags做列维度上的拼接
        temp = torch.tensor(torch.full((self.batch_size, 1), self.tag_to_ix[START_TAG]), dtype=torch.long)
        tags = torch.cat((temp, tags), dim=1)

        # 将传入的feats形状转变为[bathc_size, sequence_length, tagset_size]
        feats = feats.transpose(1, 0)
        # feats: [8, 20, 7]
        idx = 0

        # 初始化最终的结果分数张量, 每一个句子得到一个分数
        result = torch.zeros((1, self.batch_size))
        for feat_line in feats:
            # 注意: 此处区别于第三步的循环, 最重要的是这是在真实标签指导下的转移矩阵和发射矩阵的累加分数
            for i, feat in enumerate(feat_line):
                score = score + self.transitions[tags[idx][i + 1], tags[idx][i]] + feat[tags[idx][i + 1]]
            # 最后加上转移到STOP_TAG的分数
            score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[idx][-1]]
            result[0][idx] = score
            idx += 1
        return result

    # 根据传入的语句特征feats, 推断出标签序列
    def _viterbi_decode(self, feats):
        # 初始化最佳路径结果的存放列表
        result_best_path = []
        # 将输入张量变形为[batch_size, sequence_length, tagset_size]
        feats = feats.transpose(1, 0)
        # 对批次中的每一行语句进行遍历, 每个语句产生一个最优标注序列
        for feat_line in feats:
            backpointers = []
            # 初始化前向传播的张量, 设置START_TAG等于0, 约束合法序列只能从START_TAG开始
            init_vvars = torch.full((1, self.tagset_size), -10000.)
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0
            # 在第i个time_step, 张量forward_var保存第i-1个time_step的viterbi变量
            forward_var = init_vvars
            # 依次遍历i=0, 到序列最后的每一个time_step
            for feat in feat_line:
                # 保存当前time_step的回溯指针
                bptrs_t = []
                # 保存当前time_step的viterbi变量
                viterbivars_t = []
                for next_tag in range(self.tagset_size):
                    # next_tag_var[i]保存了tag_i 在前一个time_step的viterbi变量
                    # 前向传播张量forward_var加上从tag_i转移到next_tag的分数, 赋值给next_tag_var
                    # 注意此处没有加发射矩阵分数, 因为求最大值不需要发射矩阵
                    next_tag_var = forward_var + self.transitions[next_tag]

                    # 将最大的标签id加入到当前time_step的回溯列表中
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # 此处再将发射矩阵分数feat加上, 赋值给forward_var, 作为下一个time_step的前向传播张量
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                # 当前time_step的回溯指针添加进当前这一行样本的总体回溯指针中
                backpointers.append(bptrs_t)
            # 最后加上转移到STOP_TAG的分数
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            # path_score是整个路径的总得分
            path_score = terminal_var[0][best_tag_id]
            # 根据回溯指针, 解码最佳路径
            # 首先把最后一步的id值加入
            best_path = [best_tag_id]
            # 从后向前回溯最佳路径
            for bptrs_t in reversed(backpointers):
                # 通过第i个time_step得到的最佳id, 找到第i-1个time_step的最佳id
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # 将START_TAG删除
            start = best_path.pop()
            # 确认一下最佳路径中的第一个标签是START_TAG
            assert start == self.tag_to_ix[START_TAG]
            # 因为是从后向前回溯, 所以再次逆序得到总前向后的真实路径
            best_path.reverse()
            # 当前这一行的样本结果添加到最终的结果列表里
            result_best_path.append(best_path)
        return result_best_path

    # 对数似然函数的计算, 输入的是数字化编码后的语句, 和真实的标签
    # 注意: 这个函数是未来真实训练中要用到的"虚拟化的forward()"
    def neg_log_likelihood(self, sentence, tags):
        # 第一步先得到BiLSTM层的输出特征张量
        feats = self._get_lstm_features(sentence)

        # feats : [20, 8, 7] 代表一个批次有8个样本, 每个样本长度20
        # 每一个word映射到7个标签的概率, 发射矩阵

        # forward_score 代表公式推导中损失函数loss的第一项
        forward_score = self._forward_alg(feats)

        # gold_score 代表公式推导中损失函数loss的第二项
        gold_score = self._score_sentence(feats, tags)

        # 按行求和, 在torch.sum()函数值中, 需要设置dim=1 ; 同理, dim=0代表按列求和
        # 注意: 在这里, 通过forward_score和gold_score的差值来作为loss, 用来梯度下降训练模型
        return torch.sum(forward_score - gold_score, dim=1)

    # 此处的forward()真实场景是用在预测部分, 训练的时候并没有用到
    def forward(self, sentence):
        # 获取从BiLSTM层得到的发射矩阵
        lstm_feats = self._get_lstm_features(sentence)

        # 通过维特比算法直接解码最佳路径
        tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq



# 函数sentence_map完成中文文本信息的数字编码, 变成张量
def sentence_map(sentence_list, char_to_id, max_length):
    # 对一个批次的所有语句按照长短进行排序, 此步骤非必须
    sentence_list.sort(key=lambda c:len(c), reverse=True)
    # 定义一个最终存储结果特征向量的空列表
    sentence_map_list = []
    # 循环遍历一个批次内的所有语句
    for sentence in sentence_list:
        # 采用列表生成式完成字符到id的映射
        sentence_id_list = [char_to_id[c] for c in sentence]
        # 长度不够的部分用0填充
        padding_list = [0] * (max_length-len(sentence))
        # 将每一个语句向量扩充成相同长度的向量
        sentence_id_list.extend(padding_list)
        # 追加进最终存储结果的列表中
        sentence_map_list.append(sentence_id_list)
    # 返回一个标量类型值的张量
    return torch.tensor(sentence_map_list, dtype=torch.long)



# 若干辅助函数, 在类BiLSTM外部定义, 目的是辅助log_sum_exp()函数的计算
# 将Variable类型变量内部的真实值, 以python float类型返回
def to_scalar(var): # var是Variable, 维度是１
    # 返回一个python float类型的值
    return var.view(-1).data.tolist()[0]


# 获取最大值的下标
def argmax(vec):
    # 返回列的维度上的最大值下标, 此下标是一个标量float
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# 辅助完成损失函数中的公式计算
def log_sum_exp(vec): # vec是1 * 7, type是Variable
    max_score = vec[0, argmax(vec)]
    #max_score维度是1, max_score.view(1,-1)维度是1 * 1, max_score.view(1, -1).expand(1, vec.size()[1])的维度1 * 7
    # 经过expand()之后的张量, 里面所有的值都相同, 都是最大值max_score
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # vec.size()维度是1 * 7
    # 先减去max_score,最后再加上max_score, 是为了防止数值爆炸, 纯粹是代码上的小技巧
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


char_to_id = {"<PAD>":0}
# 真实标签数据, 对应为tag_to_ix中的数字标签
tag_list = [
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
# 将标签转为标量tags
tags = torch.tensor(tag_list, dtype=torch.long)

if __name__ == '__main__':
    for sentence in sentence_list:
        for _char in sentence:
            if _char not in char_to_id:
                char_to_id[_char] = len(char_to_id)
    sentence_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)
    model = BiLSTM_CRF(vocab_size=len(char_to_id), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM, \
                       hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE, \
                       sequence_length=SENTENCE_LENGTH)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(1):
        model.zero_grad()
        # 发射矩阵
        # feats = model._get_lstm_features(sentence_sequence)
        #
        # forward_score = model._forward_alg(feats)
        # print(forward_score)
        # gold_score = model._score_sentence(feats, tags)
        # print(gold_score)
        # result_tags = model._viterbi_decode(feats)
        # print(result_tags)
        loss = model.neg_log_likelihood(sentence_sequence,tags)
        print(loss)
        loss.backward()
        optimizer.step()
        result =  model(sentence_sequence)
        print(result)

