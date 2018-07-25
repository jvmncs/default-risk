import torch.nn as nn


class PooledLSTM(nn.Module):
    """A generic, concat-pooled LSTM module for use in other modules."""
    def __init__(self, n_in,
                n_out,
                dropout=0, bidirectional=False,
                pooling_type='adaptive-concat'):
        super(PooledLSTM, self).__init__()
        self.n_out = n_out
        self.lstm_cell = nn.LSTM(n_in,
                        n_out,
                        dropout=dropout, bidirectional=bidirectional)
        self.state_dim = 2 if bidirectional else 1

    def forward(self, x, h, lengths):
        x, (h, c) = self.lstm(x, h)
        return concatpool(x, lengths)

    def init_cell_state(self, batch_size):
        return nn.Parameter(
            torch.randn(self.state_dim, batch_size, self.n_out))

    @classmethod
    def concatpool(cls, ins, lengths):
        # ins is size (sequence, batch, features), lengths is size (batch,)
        nf = ins.size(-1)
        avgs = torch.sum(ins, dim=0)/lengths.view(-1, nf)
        # tried respecting var-length here, but it's wicked slow, and probably unnecessary
        maxs = nn.functional.adaptive_max_pool1d(ins.permute(1, 2, 0), 1)
        return torch.cat([ins[-1, :, :], avgs, maxs])

