import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
from sklearn.metrics import mean_squared_error


def compute_psd(data, fs=1.0):
    """计算功率谱密度（PSD），使用Welch方法。

    参数：
      - data: 输入数据（numpy数组）
      - fs: 采样频率（默认值为1.0）

    返回：
      - freqs: 对应于PSD值的频率
      - psd: 功率谱密度值
    """
    no, seq_len, dim = data.shape
    psd = np.zeros((seq_len // 2 + 1, dim))
    for i in range(dim):
        _, psd[:, i] = welch(data[:, :, i].flatten(), fs=fs, nperseg=seq_len)
    return _, np.mean(psd, axis=1)


def rmse_psd(psd_ori, psd_gen):
    """计算原始数据和生成数据PSD之间的均方根误差（RMSE）。

    参数：
      - psd_ori: 原始数据的PSD
      - psd_gen: 生成数据的PSD

    返回：
      - rmse: 均方根误差
    """
    return np.sqrt(mean_squared_error(psd_ori, psd_gen))


def kl_divergence(psd_ori, psd_gen):
    """计算原始数据和生成数据PSD之间的KL散度。

    参数：
      - psd_ori: 原始数据的PSD
      - psd_gen: 生成数据的PSD

    返回：
      - kl_div: KL散度
    """
    psd_ori += 1e-10  # 防止对数运算中的0值
    psd_gen += 1e-10
    return np.sum(psd_ori * np.log(psd_ori / psd_gen))


def visualization_psd(ori_data, generated_data, outputs_dir):
    """可视化原始数据和生成数据的PSD，并进行定量分析。

    参数：
      - ori_data: 原始数据
      - generated_data: 生成的合成数据
      - outputs_dir: 保存图片的路径
    """
    ori_data = np.array(ori_data)
    generated_data = np.array(generated_data)
    # 计算PSD
    freqs_ori, psd_ori = compute_psd(ori_data)
    freqs_gen, psd_gen = compute_psd(generated_data)

    # 绘制PSD
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_ori, psd_ori, label='原始数据', color='red')
    plt.plot(freqs_gen, psd_gen, label='合成数据', color='blue')
    plt.title('功率谱密度')
    plt.xlabel('频率')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outputs_dir, 'PSD.png'), dpi=800)
    plt.show()

    # 计算定量分析指标
    rmse = rmse_psd(psd_ori, psd_gen)
    kl_div = kl_divergence(psd_ori, psd_gen)

    # 打印结果
    print(f'均方根误差 (RMSE): {rmse:.4f}')
    print(f'KL散度: {kl_div:.4f}')
