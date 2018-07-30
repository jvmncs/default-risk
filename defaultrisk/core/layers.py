import torch
import torch.nn as nn


class PooledLSTM(nn.Module):
    """A generic, concat-pooled LSTM module for use in other modules."""
    def __init__(self, n_in,
                n_out,
                dropout=0, bidirectional=False):
        super(PooledLSTM, self).__init__()
        self.n_out = n_out
        self.lstm_cell = nn.LSTM(n_in,
                        n_out,
                        dropout=dropout, bidirectional=bidirectional)
        self.state_dim = 2 if bidirectional else 1

    def forward(self, x, h, lengths):
        x, (h, c) = self.lstm(x, h)
        return self.concatpool(x, lengths)

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
        print(ins[-1, :, :].size(), avgs.size(), maxs.size(), lengths.size())
        # TODO get last element in var length input sequence instead of last element in padded input sequence
        return torch.cat([ins[-1, :, :].unsqueeze(0), avgs.unsqueeze(0), maxs.permute(2, 0, 1)], dim=-1)
