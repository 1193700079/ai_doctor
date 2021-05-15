# 导入包
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable
# 导入之前编写好的包, 包括类, 数据集加载, 评估函数
from bilstm_crf import BiLSTM_CRF
from loader_data import load_dataset
from evaluate_model import evaluate

# 训练模型的函数
def train(data_loader, data_size, batch_size, embedding_dim, hidden_dim,
          sentence_length, num_layers, epochs, learning_rate, tag2id,
          model_saved_path, train_log_path,
          validate_log_path, train_history_image_path):
    '''
    data_loader: 数据集的加载器, 之前已经通过load_dataset完成了构造
    data_size:   训练集和测试集的样本数量
    batch_size:  批次的样本个数
    embedding_dim:  词嵌入的维度
    hidden_dim:     隐藏层的维度
    sentence_length:  文本限制的长度
    num_layers:       神经网络堆叠的LSTM层数
    epochs:           训练迭代的轮次
    learning_rate:    学习率
    tag2id:           标签到id的映射字典
    model_saved_path: 模型保存的路径
    train_log_path:   训练日志保存的路径
    validate_log_path:  测试集日志保存的路径
    train_history_image_path:  训练数据的相关图片保存路径
    '''
    # 将中文字符和id的对应码表加载进内存
    char2id = json.load(open("data/char_to_id.json", mode="r", encoding="utf-8"))
    # 初始化BiLSTM_CRF模型
    model = BiLSTM_CRF(vocab_size=len(char2id), tag_to_ix=tag2id,
                   embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                   batch_size=batch_size, num_layers=num_layers,
                   sequence_length=sentence_length)

    # 定义优化器, 使用SGD作为优化器(pytorch中Embedding支持的GPU加速为SGD, SparseAdam）
    # 参数说明如下:
    # lr:          优化器学习率
    # momentum:    优化下降的动量因子, 加速梯度下降过程
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.85)

    # 设定优化器学习率更新策略
    # 参数说明如下:
    # optimizer:    优化器
    # step_size:    更新频率, 每过多少个epoch更新一次优化器学习率
    # gamma:        学习率衰减幅度,
    #               按照什么比例调整(衰减)学习率(相对于上一轮epoch), 默认0.1
    #   例如:
    #   初始学习率 lr = 0.5,    step_size = 20,    gamma = 0.1
    #              lr = 0.5     if epoch < 20
    #              lr = 0.05    if 20 <= epoch < 40
    #              lr = 0.005   if 40 <= epoch < 60
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.2)

    # 初始化存放训练中损失, 准确率, 召回率, F1等数值指标
    train_loss_list = []
    train_acc_list = []
    train_recall_list = []
    train_f1_list = []
    train_log_file = open(train_log_path, mode="w", encoding="utf-8")
    # 初始化存放测试中损失, 准确率, 召回率, F1等数值指标
    validate_loss_list = []
    validate_acc_list = []
    validate_recall_list = []
    validate_f1_list = []
    validate_log_file = open(validate_log_path, mode="w", encoding="utf-8")
    # 利用tag2id生成id到tag的映射字典
    id2tag = {v:k for k, v in tag2id.items()}
    # 利用char2id生成id到字符的映射字典
    id2char = {v:k for k, v in char2id.items()}

    # 按照参数epochs的设定来循环epochs次
    for epoch in range(epochs):
        # 在进度条打印前, 先输出当前所执行批次
        tqdm.write("Epoch {}/{}".format(epoch + 1, epochs))
        # 定义要记录的正确总实体数, 识别实体数以及真实实体数
        total_acc_entities_length, \
        total_predict_entities_length, \
        total_gold_entities_length = 0, 0, 0
        # 定义每batch步数, 批次loss总值, 准确度, f1值
        step, total_loss, correct, f1 = 1, 0.0, 0, 0

        # 开启当前epochs的训练部分
        for inputs, labels in tqdm(data_loader["train"]):
            # 将数据以Variable进行封装
            inputs, labels = Variable(inputs), Variable(labels)
            # 在训练模型期间, 要在每个样本计算梯度前将优化器归零, 不然梯度会被累加
            optimizer.zero_grad()
            # 此处调用的是BiLSTM_CRF类中的neg_log_likelihood()函数
            loss = model.neg_log_likelihood(inputs, labels)
            # 获取当前步的loss, 由tensor转为数字
            step_loss = loss.data
            # 累计每步损失值
            total_loss += step_loss
            # 获取解码最佳路径列表, 此时调用的是BiLSTM_CRF类中的forward()函数
            best_path_list = model(inputs)
            # 模型评估指标值获取包括:当前批次准确率, 召回率, F1值以及对应的实体个数
            step_acc, step_recall, f1_score, acc_entities_length, \
            predict_entities_length, gold_entities_length = evaluate(inputs.tolist(),
                                                                     labels.tolist(),
                                                                     best_path_list,
                                                                     id2char,
                                                                     id2tag)
            # 训练日志内容
            log_text = "Epoch: %s | Step: %s " \
                       "| loss: %.5f " \
                       "| acc: %.5f " \
                       "| recall: %.5f " \
                       "| f1 score: %.5f" % \
                       (epoch, step, step_loss, step_acc, step_recall,f1_score)
            # 分别累计正确总实体数、识别实体数以及真实实体数
            total_acc_entities_length += acc_entities_length
            total_predict_entities_length += predict_entities_length
            total_gold_entities_length += gold_entities_length

            # 对损失函数进行反向传播
            loss.backward()
            # 通过optimizer.step()计算损失, 梯度和更新参数
            optimizer.step()
            # 记录训练日志
            train_log_file.write(log_text + "\n")
            step += 1
        # 获取当前epochs平均损失值(每一轮迭代的损失总值除以总数据量)
        epoch_loss = total_loss / data_size["train"]
        # 计算当前epochs准确率
        total_acc = total_acc_entities_length / total_predict_entities_length
        # 计算当前epochs召回率
        total_recall = total_acc_entities_length / total_gold_entities_length
        # 计算当前epochs的F1值
        total_f1 = 0
        if total_acc + total_recall != 0:
            total_f1 = 2 * total_acc * total_recall / (total_acc + total_recall)
        log_text = "Epoch: %s " \
                   "| mean loss: %.5f " \
                   "| total acc: %.5f " \
                   "| total recall: %.5f " \
                   "| total f1 scroe: %.5f" % (epoch, epoch_loss,
                                               total_acc,
                                               total_recall,
                                               total_f1)

        # 当前epochs训练后更新学习率, 必须在优化器更新之后
        scheduler.step()

        # 记录当前epochs训练loss值(用于图表展示), 准确率, 召回率, f1值
        train_loss_list.append(epoch_loss)
        train_acc_list.append(total_acc)
        train_recall_list.append(total_recall)
        train_f1_list.append(total_f1)
        train_log_file.write(log_text + "\n")


        # 定义要记录的正确总实体数, 识别实体数以及真实实体数
        total_acc_entities_length, \
        total_predict_entities_length, \
        total_gold_entities_length = 0, 0, 0
        # 定义每batch步数, 批次loss总值, 准确度, f1值
        step, total_loss, correct, f1 = 1, 0.0, 0, 0

        # 开启当前epochs的验证部分
        for inputs, labels in tqdm(data_loader["validation"]):
            # 将数据以Variable进行封装
            inputs, labels = Variable(inputs), Variable(labels)
            # 此处调用的是BiLSTM_CRF类中的neg_log_likelihood 函数
            # 返回最终的CRF的对数似然结果
            loss = model.neg_log_likelihood(inputs, labels)
            # 获取当前步的loss值, 由tensor转为数字
            step_loss = loss.data
            # 累计每步损失值
            total_loss += step_loss
            # 获取解码最佳路径列表, 此时调用的是BiLSTM_CRF类中的forward()函数
            best_path_list = model(inputs)
            # 模型评估指标值获取: 当前批次准确率, 召回率, F1值以及对应的实体个数
            step_acc, step_recall, f1_score, acc_entities_length, \
            predict_entities_length, gold_entities_length = evaluate(inputs.tolist(),
                                                                     labels.tolist(),
                                                                     best_path_list,
                                                                     id2char,
                                                                     id2tag)

            # 训练日志内容
            log_text = "Epoch: %s | Step: %s " \
                       "| loss: %.5f " \
                       "| acc: %.5f " \
                       "| recall: %.5f " \
                       "| f1 score: %.5f" % \
                       (epoch, step, step_loss, step_acc, step_recall,f1_score)
            # 分别累计正确总实体数、识别实体数以及真实实体数
            total_acc_entities_length += acc_entities_length
            total_predict_entities_length += predict_entities_length
            total_gold_entities_length += gold_entities_length

            # 记录验证集损失日志
            validate_log_file.write(log_text + "\n")
            step += 1

        # 获取当前批次平均损失值(每一批次损失总值除以数据量)
        epoch_loss = total_loss / data_size["validation"]
        # 计算总批次准确率
        total_acc = total_acc_entities_length / total_predict_entities_length
        # 计算总批次召回率
        total_recall = total_acc_entities_length / total_gold_entities_length
        # 计算总批次F1值
        total_f1 = 0
        if total_acc + total_recall != 0:
            total_f1 = 2 * total_acc * total_recall / (total_acc + total_recall)
        log_text = "Epoch: %s " \
                   "| mean loss: %.5f " \
                   "| total acc: %.5f " \
                   "| total recall: %.5f " \
                   "| total f1 scroe: %.5f" % (epoch, epoch_loss,
                                               total_acc,
                                               total_recall,
                                               total_f1)

        # 记录当前批次验证loss值(用于图表展示)准确率, 召回率, f1值
        validate_loss_list.append(epoch_loss)
        validate_acc_list.append(total_acc)
        validate_recall_list.append(total_recall)
        validate_f1_list.append(total_f1)
        validate_log_file.write(log_text + "\n")


    # 保存模型
    torch.save(model.state_dict(), model_saved_path)

    # 将loss下降历史数据转为图片存储
    save_train_history_image(train_loss_list,
                             validate_loss_list,
                             train_history_image_path,
                             "Loss")
    # 将准确率提升历史数据转为图片存储
    save_train_history_image(train_acc_list,
                             validate_acc_list,
                             train_history_image_path,
                             "Acc")
    # 将召回率提升历史数据转为图片存储
    save_train_history_image(train_recall_list,
                             validate_recall_list,
                             train_history_image_path,
                             "Recall")
    # 将F1上升历史数据转为图片存储
    save_train_history_image(train_f1_list,
                             validate_f1_list,
                             train_history_image_path,
                             "F1")
    print("train Finished".center(100, "-"))


# 按照传入的不同路径, 绘制不同的训练曲线
def save_train_history_image(train_history_list,
                             validate_history_list,
                             history_image_path,
                             data_type):
    # 根据训练集的数据列表, 绘制折线图
    plt.plot(train_history_list, label="Train %s History" % (data_type))
    # 根据测试集的数据列表, 绘制折线图
    plt.plot(validate_history_list, label="Validate %s History" % (data_type))
    # 将图片放置在最优位置
    plt.legend(loc="best")
    # 设置x轴的图标为轮次Epochs
    plt.xlabel("Epochs")
    # 设置y轴的图标为参数data_type
    plt.ylabel(data_type)
    # 将绘制好的图片保存在特定的路径下面, 并修改图片名字中的"plot"为对应的data_type
    plt.savefig(history_image_path.replace("plot", data_type))
    plt.close()


# 参数1:批次大小
BATCH_SIZE = 8
# 参数2:训练数据文件路径
train_data_file_path = "data/train.npz"
# 参数3:加载 DataLoader 数据
data_loader, data_size = load_dataset(train_data_file_path, BATCH_SIZE)
# 参数4:记录当前训练时间（拼成字符串用）
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
# 参数5:标签码表对照
tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}
# 参数6:训练文件存放路径
model_saved_path = "model/bilstm_crf_state_dict_%s.pt" % (time_str)
# 参数7:训练日志文件存放路径
train_log_path = "log/train_%s.log" % (time_str)
# 参数8:验证打印日志存放路径
validate_log_path = "log/validate_%s.log" % (time_str)
# 参数9:训练历史记录图存放路径
train_history_image_path = "log/bilstm_crf_train_plot_%s.png" % (time_str)
# 参数10:字向量维度
EMBEDDING_DIM = 200
# 参数11:隐层维度
HIDDEN_DIM = 100
# 参数12:句子长度
SENTENCE_LENGTH = 20
# 参数13:堆叠 LSTM 层数
NUM_LAYERS = 1
# 参数14:训练批次
EPOCHS = 100
# 参数15:初始化学习率
LEARNING_RATE = 0.5


if __name__ == '__main__':
    train(data_loader, data_size, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, SENTENCE_LENGTH,
          NUM_LAYERS, EPOCHS, LEARNING_RATE, tag_to_id,
          model_saved_path, train_log_path, validate_log_path, train_history_image_path)