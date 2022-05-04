import torch
from torch import nn

class Conv_lstm(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, kernel_size, input_num, target_num, dropout,
                device=torch.device('cuda:0')):
        super().__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.input_num = input_num
        self.kernel_size = kernel_size

        self.c1d = nn.Conv1d(in_channels=self.input_num, out_channels=n_features, kernel_size = self.kernel_size, stride = 1) # 1D CNN layer

        self.norm1d = nn.BatchNorm1d(n_features)
        self.act_fc = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout, 
            batch_first=True
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=target_num)
        
    def forward(self, sequences):
        sequences = self.c1d(sequences.transpose(1,2))
        sequences = self.act_fc(sequences)
        sequences = self.norm1d(sequences)
        lstm_in = sequences.transpose(1,2)
        lstm_out, _ = self.lstm(lstm_in)
        last_time_step = lstm_out[:,-1,:]
        y_pred = self.linear(last_time_step)
        return y_pred


class lstm(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, input_num, target_num, dropout,
                device=torch.device('cuda:0')):
        super().__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.input_num = input_num

        self.act_fc = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.input_num)

        self.lstm = nn.LSTM(
            input_size=self.input_num,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout, 
            batch_first=True
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=target_num)
    
    def forward(self, sequences):
        sequences = self.layer_norm(sequences)
        lstm_out, _ = self.lstm(sequences)
        last_time_step = lstm_out[:,-1,:]
        last_time_step = self.act_fc(last_time_step)
        y_pred = self.linear(last_time_step)
        return y_pred

class Bilstm(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, input_num, target_num, dropout,
                device=torch.device('cuda:0')):
        super().__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.input_num = input_num

        self.act_fc = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.input_num)

        self.bilstm = nn.LSTM(
            input_size=self.input_num,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout= dropout, 
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(in_features=n_hidden * 2, out_features=target_num)
    
    def forward(self, sequences):
        sequences = self.layer_norm(sequences)
        lstm_out, _ = self.bilstm(sequences)
        last_time_step = lstm_out[:,-1,:]
        last_time_step = self.act_fc(last_time_step)
        y_pred = self.linear(last_time_step)
        return y_pred
