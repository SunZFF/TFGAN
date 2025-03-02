import torch.nn as nn
import torch.fft as fft
import numpy as np
import torch


def get_rnn_cell(module_name):
    """Basic RNN Cell.
      Args:
        - module_name: gru, lstm
      Returns:
        - rnn_cell: RNN Cell
    """
    assert module_name in ['gru', 'lstm']
    rnn_cell = None#确保传入的是'gru',或'lstm'，否则停止运行
    # GRU
    if module_name == 'gru':
        rnn_cell = nn.GRU
    # LSTM
    elif module_name == 'lstm':
        rnn_cell = nn.LSTM
    return rnn_cell


class Embedder(nn.Module):
    """Embedding network between original feature space to latent space.
        Args:
          - input: input time-series features. Size:(Num, Len, Dim) = (3661, 24, 6)
        Returns:
          - H: embedding features size: (Num, Len, Dim) = (3661, 24, 6)
        """

    def __init__(self, para):
        super(Embedder, self).__init__()
        rnn_cell = get_rnn_cell(para['module'])
        self.rnn = rnn_cell(input_size=para['input_dim'], hidden_size=para['hidden_dim'], num_layers=para['num_layer'],
                            batch_first=True)#rnn层
        self.fc = nn.Linear(para['hidden_dim'], para['hidden_dim'])#全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        e_outputs, _ = self.rnn(X)
        H = self.fc(e_outputs)
        H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    """Recovery network from latent space to original space.
    Args:
      - H: latent representation
      - T: input time information
    Returns:
      - X_tilde: recovered data
    """

    def __init__(self, para):
        super(Recovery, self).__init__()
        rnn_cell = get_rnn_cell(para['module'])
        self.rnn = rnn_cell(input_size=para['hidden_dim'], hidden_size=para['input_dim'], num_layers=para['num_layer'],
                            batch_first=True)
        self.fc = nn.Linear(para['input_dim'], para['input_dim'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        r_outputs, _ = self.rnn(H)
        X_tilde = self.fc(r_outputs)
        X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.
    Args:
      - Z: random variables
    Returns:
      - E: generated embedding
    """

    def __init__(self, para):
        super(Generator, self).__init__()
        rnn_cell = get_rnn_cell(para['module'])
        self.rnn = rnn_cell(input_size=para['input_dim'], hidden_size=para['hidden_dim'], num_layers=para['num_layer'],
                            batch_first=True)
        self.fc = nn.Linear(para['hidden_dim'], para['hidden_dim'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, Z):
        g_outputs, _ = self.rnn(Z)
        E = self.fc(g_outputs)
        E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.
    Args:
      - H: latent representation
      - T: input time information
    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """

    def __init__(self, para):
        super(Supervisor, self).__init__()
        rnn_cell = get_rnn_cell(para['module'])
        self.rnn = rnn_cell(input_size=para['hidden_dim'], hidden_size=para['hidden_dim'], num_layers=para['num_layer'] - 1,
                            batch_first=True)
        self.fc = nn.Linear(para['hidden_dim'], para['hidden_dim'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        s_outputs, _ = self.rnn(H)
        S = self.fc(s_outputs)
        S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.
    Args:
      - H: latent representation
      - T: input time information
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """

    def __init__(self, para):
        super(Discriminator, self).__init__()
        rnn_cell = get_rnn_cell(para['module'])
        self.rnn = rnn_cell(input_size=para['hidden_dim'], hidden_size=para['hidden_dim'], num_layers=para['num_layer'],
                            batch_first=True)
        self.rnn2 = rnn_cell(input_size=para['input_dim'], hidden_size=para['hidden_dim'], num_layers=para['num_layer'],
                            batch_first=True)
        self.fc = nn.Linear(para['hidden_dim'],para['hidden_dim']  )
        self.sigmoid= nn.Sigmoid()
        self.hidden=para['hidden_dim']
        self.rwight=para['resampled_fft_weight']

    def forward(self, H, r_data):
        H_freq = fft.fft(H,dim=1).abs()
        r_data_freq = fft.fft(r_data,dim=0).abs()
        d_outputs1, _ = self.rnn(H)
        d_outputs2, _ = self.rnn(H_freq)
        d_outputs3, _ = self.rnn2(r_data_freq)
        d_outputs2 = d_outputs2 * (1-self.rwight)+ d_outputs3 * self.rwight
        d_outputs1 = self.fc(d_outputs1)
        d_outputs2 = self.fc(d_outputs2)
        d_outputs = torch.abs(d_outputs1-d_outputs2)#差分
        d_outputs=torch.sum(d_outputs, dim=-1, keepdim=True)
        d_outputs=d_outputs/self.hidden
        Y= self.sigmoid(d_outputs)



        return Y
