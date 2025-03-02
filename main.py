import argparse
import numpy as np

# 1. Training model
from run import train, test
# 2. Data loading
from data_loading import real_data_loading


def main(opt):
    # Data loading
    ori_data , r_data = real_data_loading(opt.data_dir, opt.data_name, opt.seq_len, opt.resampled_interval )
    r_data_array = np.array(r_data)
    r_data = np.mean(r_data_array, axis=0)
    print(opt.data_name + ' dataset is ready.')

    # Training or Testing
    if opt.is_test:
        test(opt, ori_data, r_data)
    else:
        train(opt, ori_data, r_data)
        test(opt, ori_data, r_data)


if __name__ == '__main__':
    """Main function for timeGAN experiments.
    Args:
      - data_name: 'stock', 'ex_rate', 'etth1', 'sine'
      - seq_len: sequence length
      - Network parameters (should be optimized for different datasets)
        - module: gru, lstm
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
        - resampled_fft_weight：weight for resampled data after FFT
        - H_hat_weight：weight for Generator output data
      - metric_iteration: number of iterations for metric computation  、、



    Returns:
      - ori_data: original data
      - gen_data: generated synthetic data
      - metric_results: discriminative and predictive scorqves
    """
    # Args for the main function
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('--data_name', type=str, default='sine', choices=['stock', 'ex_rate', 'etth1', 'sine_long', 'sine'], help='dataset name')
    parser.add_argument('--seq_len', type=int, default=24, help='sequence length')
    # Network parameters (should be optimized for different datasets)
    parser.add_argument('--module', choices=['gru', 'lstm'], default='gru', type=str)
    parser.add_argument('--hidden_dim', type=int, default=6, help='hidden state dimensions')
    parser.add_argument('--num_layer', type=int, default=3, help='number of layers')
    # Model training and testing parameters`
    parser.add_argument('--gamma', type=float, default=1, help='gamma weight for G_loss and D_loss')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--iterations', type=int, default=20000, help='Training iterations')
    parser.add_argument('--print_times', type=int, default=10, help='Print times when Training')
    parser.add_argument('--batch_size', type=int, default=128, help='the number of samples in mini-batch')
    parser.add_argument('--synth_size', type=int, default=0, help='the number of samples in synthetic data, '
                                                                  '0--len(ori_data)')
    parser.add_argument('--metric_iteration', type=int, default=10, help='iterations of the metric computation')
    # Save and Load
    parser.add_argument('--data_dir', type=str, default="./data", help='path to stock and energy data')
    parser.add_argument('--networks_dir', type=str, default="./trained_networks", help='path to checkpoint')
    parser.add_argument('--output_dir', type=str, default="./output", help='folder to output metrics and images')
    # Model running parameters
    parser.add_argument('--is_test', type=bool, default=False, help='iterations of the metric computation')
    parser.add_argument('--only_visualize_metric', type=bool, default=False, help='only compute visualization metrics')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='load pretrain networks')
    #提取长期的趋势信息
    parser.add_argument('--resampled_fft_weight', type=float, default=0.0, help='weight for resampled data after FFT')#重新采样后数据的权重
    parser.add_argument('--resampled_interval', type=int, default=20, help='sampling interval for resampled data')#重新采样的采样间隔
    #对输出微调
    #parser.add_argument('--H_hat_weight', type=float, default=1.0, help='weight for Generator output data')

    # Call main function
    opt = parser.parse_args()


    main(opt)
