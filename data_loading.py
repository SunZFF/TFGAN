import numpy as np
import os
import pandas as pd

def MinMaxScaler(data):#归一化
    """Min Max normalizer.
    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def real_data_loading(data_dir, data_name, seq_len,sampling_interval):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: stock or energy
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """

    assert data_name in ['stock', 'ex_rate', 'etth1', 'sine_long', 'sine']

    ori_data = []
    if data_name == 'stock':
        ori_data = np.loadtxt(os.path.join(data_dir, 'stock_data.csv'), delimiter=",", skiprows=1)
    elif data_name == 'ex_rate':
        ori_data = np.loadtxt(os.path.join(data_dir, 'exchange_rate.csv'), delimiter=",", skiprows=1)
    elif data_name == 'etth1':
        ori_data = np.loadtxt(os.path.join(data_dir, 'ETTh1_data.csv'), delimiter=",", skiprows=1)
    elif data_name == 'sine_long':
        ori_data = np.loadtxt(os.path.join(data_dir, 'sine_long_data.csv'), delimiter=",", skiprows=1)
    elif data_name == 'sine':
        ori_data = np.loadtxt(os.path.join(data_dir, 'sine_data.csv'), delimiter=",", skiprows=0)

    # 反转数据顺序
    ori_data = ori_data[::-1]
    # 归一化数据
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # 打乱数据
    idx = np.random.permutation(len(temp_data))#生成一个随机排列的索引数组
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    r_data = []

    max_sampling_interval = len(ori_data) // seq_len
    if (sampling_interval <= max_sampling_interval):
        sampling_num = (len(ori_data) // sampling_interval) // seq_len#计算采样次数
        for i in range(sampling_num):#滑动采样
            start_idx = i * sampling_interval
            end_idx = start_idx + seq_len
            if end_idx <= len(ori_data):
                r_data.append(ori_data[start_idx:end_idx])
    else:
        r_data = ori_data[::sampling_interval]

    return data, r_data
