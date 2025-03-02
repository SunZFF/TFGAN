import numpy as np


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # 分割训练集和测试集（原始数据）
    no = len(data_x)
    idx = np.random.permutation(no)#生成一个随机排列的索引数组
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # 分割训练集和测试集 (生成数据)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def MinMaxScaler(data):#归一化
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
    Args:
      - data: original data
    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))#每个序列中最大序列长度
        time.append(len(data[i][:, 0]))#将每个序列的长度添加到time中

    return time, max_seq_len


def random_generator(batch_size, z_dim, max_seq_len, *T):
    """Random vector generation.
    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length
    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        if not T:#如果未提供时间信息T，生成一个形状为 (max_seq_len, z_dim)的随机向量序列，元素取值[0,1)之间
            temp = np.random.uniform(0., 1, [max_seq_len, z_dim])
        else:#如果提供了时间信息T，生成一个形状为 (T, z_dim)的随机向量序列，T为可变长度，元素取值[0,1)之间
            T_mb = T[0]
            temp = np.random.uniform(0., 1, [T_mb[i], z_dim])
        Z_mb.append(temp)
    return Z_mb


def batch_generator(data, time, batch_size):#生成batch_size数据
    """Mini-batch generator.
    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch
    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb
