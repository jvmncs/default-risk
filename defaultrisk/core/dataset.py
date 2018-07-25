import torch
from torch.utils.data import Dataset

# After collating, Dimensions of each instance in RiskDataset
# for each Module in Risk model should be:
#   Application:            (a, na) where a is batch size (num current apps), na is number of features
#   Bureau:                 (b, a, nb) where b is num bureau apps, nb is num features per bureau app
#   BureauBal:              (a, bb, b, nbb) where bb is num balances per bureau app, nbb is num features per balance
#   PreviousApplication:    (p, a, np) where p is num prev apps, np is num features per prev app
#   POSCashBalance:         (a, pc, p, npc) where pc is num pos balances per prev app, npc is num features per balance
#   InstallmentsPayments:   (a, ip, p, nip) where ip is num payments per prev app, nip is num features per payment
#   CreditCardBalance:      (a, c, p, nc) where c is num cc balances per prev app, nc is num features per cc balance


# TODO: create a custom Sampler for each non-root Dataset that collates according to it's child(ren)'s index map,
#   e.g. for an application with index i, sampling from BureauDataset should return all (b, nb)-shaped tensors x_j
#   such that j is in the list self.app.bureau_map[i], concatenated along newly unsqueezed dimension 1

class RiskDataset(Dataset):
    """Prepares data from all tables for input into the Risk model.

    Arguments are the respective tables' Dataset objects.
    If the Dataset isn't a leaf node in the Dataset tree, it comes with a
        map that designates which of its indices maps to which indices in
        its child(ren) Dataset(s).
    """
    def __init__(self, app, bureau, burbal, prvapp, pos, instpay, ccbal):
        super(RiskDataset, self).__init__()
        self.app = app
        self.bureau = bureau
        self.burbal = burbal
        self.prvapp = prvapp
        self.pos = pos
        self.instpay = instpay
        self.ccbal = ccbal
        
    def __getitem__(self, ix):
        """Implements obligatory Dataset functionality of retrieving a single instance.

        In our case, a single instance consists of a single applicant, along with
        all other sequences from other tables that are related to that applicant.
        """
        return # TODO

    def __len__(self):
        """Obligatory Dataset __len__ magic method.

        In this case, it's ApplicationDataset.__len__.
        """
        return len(self.app)

class ApplicationDataset(Dataset):
    """Prepares data from the application{train/test}.csv table."""
    def __init__(self, filepath, bureau_map, prevapp_map):
        super(ApplicationDataset, self).__init__()
        self.data = torch.randn((1000, 125), dtype=torch.float) # (a, na) # TODO change to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
        self.bureau_map = bureau_map # maps each index in the `data` iterable to a collection of indices in the BureauDataset
        self.prvapp_map = prvapp_map # maps each index in the `data` iterable to a collection of indices in the PreviousApplicationDataset
        
    def __getitem__(self, ix):
        return self.data[ix]

    def __len__(self):
        return len(self.data)

class BureauDataset(Dataset):
    """Prepares data from the bureau.csv table"""
    def __init__(self, filepath, burbal_map):
        super(BureauDataset, self).__init__()
        self.data = torch.randn((15, 1000, 10), dtype=torch.float) # (b, a, nb) # TODO change to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
        self.burbal_map = burbal_map # maps each index in the `data` iterable to a collection of indices in the BureauBalanceDataset

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)

class BureauBalanceDataset(Dataset):
    """Prepares data from the bureau_balance.csv table"""
    def __init__(self):
        super(BureauBalanceDataset, self).__init__()
        self.data = torch.randn((100, 15 * 1000, 8), dtype=torch.float) # (bb, b * a, nbb) # TODO change to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
        # TODO should the above be 4d (1000, 100, 15, 8) or 3d (100, 15 * 1000, 8)?

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return len(self.data)
