from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime


def visualization(ori_data, generated_data, analysis, outputs_dir):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
      - outputs_dir: path to images output
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])#确定要分析的样本数，1000和原始样本数取最小
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]#随机选择样本生成索引

    # Data preprocessing
    ori_data = np.asarray(ori_data)#转换为numpy数组
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[:anal_sample_no]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):#计算平均值，并将结果连接到先前计算的结果数组中
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")#pca_results[:, 0]：pca降维后的第一个维度
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")#pca_results[:, 1]：pca降维后的第二个维度

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        # 添加时间戳到文件名以避免覆盖
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'PCA_{timestamp}.png'

        plt.savefig(os.path.join(outputs_dir, file_name), dpi=800)
        plt.show()


    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        # 添加时间戳到文件名以避免覆盖
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f't-SNE_{timestamp}.png'

        plt.savefig(os.path.join(outputs_dir, file_name), dpi=800)
        plt.show()
