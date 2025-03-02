import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator
import torch
import torch.nn as nn


class Discriminator(nn.Module):#后验鉴别器

    def __init__(self, input_dim, hidden_dim, num_layer):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        d_outputs, d_last_states = self.rnn(X)
        y_hat_logit = self.fc(d_last_states)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


def discriminative_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置最大序列长度
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)#划分训练集和测试集

    #建立后验RNN鉴别器
    # Network-parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    discriminator = Discriminator(dim, hidden_dim, 1).to(device)
    optim_discriminator = torch.optim.Adam(discriminator.parameters())#使用优化器
    loss_function = nn.BCEWithLogitsLoss()

    # Training step
    for itt in range(iterations):
        discriminator.train()
        optim_discriminator.zero_grad()#梯度清零
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)#生成一个批次的训练数据
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        # Forward
        X_mb = torch.tensor(X_mb, dtype=torch.float32).to(device)#将数据转换为张量并移至设备
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32).to(device)
        y_logit_real, y_pred_real = discriminator(X_mb)#logit表示激活函数之前的原始输出值，pred表示真正的预测输出值
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)
        # Loss function
        d_loss_real = torch.mean(loss_function(y_logit_real, torch.ones_like(y_logit_real)))#loss_function交叉熵损失函数，torch.mean():取均值函数
        d_loss_fake = torch.mean(loss_function(y_logit_fake, torch.zeros_like(y_logit_fake)))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optim_discriminator.step()#执行一步优化器的优化操作

    # ------ Test the performance on the testing set
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)#将测试集转换为张量并移动到设备
    _, y_pred_real_curr = discriminator(test_x)#获取判别器对真实数据预测概率
    y_pred_real_curr = y_pred_real_curr.cpu().detach().numpy()[0]#将张量转换为Numpy数组

    test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32).to(device)#将测试集转换为张量并移动到设备
    _, y_pred_fake_curr = discriminator(test_x_hat)#获取判别器生成数据的预测概率
    y_pred_fake_curr = y_pred_fake_curr.cpu().detach().numpy()[0]#将张量转换为Numpy数组
    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))#拼接数据
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])),
                                   axis=0)#拼接数据
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))#(y_pred_final > 0.5)表示>0.5则预测类别为1
    discriminative_score = np.abs(0.5 - acc)#acc越小，表示生成的数据越接近真实数据；0.5表示随机猜测的概率，与随机猜测的概率偏差越小，证明性能越好
    print('discriminative_score: ', discriminative_score)

    return discriminative_score
