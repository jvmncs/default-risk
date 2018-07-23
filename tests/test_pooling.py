from defaultrisk.core.layers import concatpool

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import time

dev = 'cpu' if not torch.cuda.is_available() else 'cuda'

seq_lens = [12, 8, 4]
seqs, max_seqs = [], []
true_mean, true_max = [], []
naive_pooled, naive_pooled_max = [], []
true_pooled, true_pooled_max = [], []

max_len = max(seq_lens)
for n in seq_lens:
    # Make alternating series, `sum((n+1)*(-1)^n)`
    alternator = torch.tensor([1,-1] * (n // 2), dtype=torch.float, device=dev)
    x = (torch.arange(n, dtype=torch.float, device=dev) + 1) * alternator
    seqs.append(x)

    y = (torch.arange(n, dtype=torch.float, device=dev) + 1) * -1
    max_seqs.append(y)

    # Store and compute true global mean
    true_mean.append(torch.mean(x))
    true_max.append(torch.max(y))


# Pad the sequences
padded = pad_sequence(seqs).unsqueeze(-1) # output is (sequence, batch, features)
padded_max = pad_sequence(max_seqs).unsqueeze(-1)

# Naive average pooling
time0 = time.time()
for i in range(100000):
    nap = nn.functional.adaptive_avg_pool1d(padded.permute(1, 2, 0), 1) # input is (batch, features, sequence)
naive_time = time.time() - time0

# Variable-length average pooling
lengths = torch.tensor(seq_lens, dtype=torch.float, device=dev)
time1 = time.time()
for i in range(100000):
    tap = torch.sum(padded, dim=0)/lengths.view(-1,1)
truepool_time = time.time() - time1

# TODO test concat pooling here

for i in range(len(seqs)):
    assert true_mean[i].item() == -.5
    assert true_max[i].item() == - 1
    assert true_mean[i].item() == tap[i].squeeze().item()
    if i != 0:
        assert true_mean[i].item() != nap[i].squeeze().item()
    else:
        assert true_mean[i].item() == nap[i].squeeze().item()
    print(true_mean[i].item(), tap[i].squeeze().item(), nap[i].squeeze().item())
    assert truepool_time < 1.2 * naive_time
