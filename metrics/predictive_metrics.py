"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time, train_test_divide


class Predictor(nn.Module):#后验预测器
    def __init__(self, input_dim, hidden_dim, num_layer):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        p_outputs, p_last_states = self.rnn(X)
        y_hat_logit = self.fc(p_outputs)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置最大序列长度
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # 建立RNN后验预测器
    # Network-parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    predictor = Predictor(dim-1, hidden_dim, 1).to(device)  # why dim-1 rather than dim, puzzle for me(AlanDongMu)
    optim_predictor = torch.optim.Adam(predictor.parameters())
    loss_function = nn.L1Loss()

    # Training step
    for itt in range(iterations):
        predictor.train()
        optim_predictor.zero_grad()
        # Batch setting
        idx = np.random.permutation(len(generated_data))#生成一个长度为generated_data的随机数组
        train_idx = idx[:batch_size]#训练索引

        X_mb = list(generated_data[i][:-1, :(dim - 1)] for i in train_idx)#提取训练集样本
        T_mb = list(generated_time[i] - 1 for i in train_idx)#提取时间步
        Y_mb = list(
            np.reshape(generated_data[i][1:, (dim - 1)], [len(generated_data[i][1:, (dim - 1)]), 1]) for i in train_idx)#提取标签
        # Forward
        X_mb = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Y_mb = torch.tensor(Y_mb, dtype=torch.float32).to(device)
        y_pred = predictor(X_mb)
        # Loss for the predictor
        p_loss = loss_function(y_pred, Y_mb)#使得生成数据尽可能接近其标签

        p_loss.backward()
        optim_predictor.step()

    idx = np.random.permutation(len(ori_data))#生成一个长度为ori_data的随机数组
    train_idx = idx[:no]#选择前no个索引值作为索引

    X_mb = list(ori_data[i][:-1, :(dim - 1)] for i in train_idx)#提取训练集样本
    T_mb = list(ori_time[i] - 1 for i in train_idx)#提取时间步
    Y_mb = list(np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1]) for i in train_idx)#提取标签

    # Prediction
    X_mb = torch.tensor(X_mb, dtype=torch.float32).to(device)
    pred_Y_curr = predictor(X_mb)#预测
    pred_Y_curr = pred_Y_curr.cpu().detach().numpy()#转换为Numpy数组

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])#计算平均绝对误差，后求和

    predictive_score = MAE_temp / no
    print('predictive_score: ', predictive_score)
    return predictive_score
