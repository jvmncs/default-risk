import torch
from torch.utils.data import Dataset

# For reference, the following classes form a tree of Datasets, mirroring the tree-like relationship
# of the CSVs, as shown in the diagram here https://www.kaggle.com/c/home-credit-default-risk/data
# Specifically, the tree is as follows:
#   |ApplicationDataset
#   |---|BureauDataset
#   |---|---|BureauBalanceDataset
#   |---|PreviousApplicationsDataset
#   |---|---|POSCashBalanceDataset
#   |---|---|InstallmentsPaymentsDataset
#   |---|---|CreditCardBalanceDataset

# After collating, an instance in RiskDataset
# should have a specific tensor for each Module in our Risk model.
#   Application:            x_app, of size (a, na), where a is batch size (num current apps), na is number of features
#   Bureau:                 x_bureau, of size (b, a, nb), where b is num bureau apps, nb is num features per bureau app
#   BureauBalance:          x_burbal, of size (a, bb, b, nbb), where bb is num balances per bureau app, nbb is num features per balance
#   PreviousApplication:    x_prvapp, of size (p, a, np), where p is num prev apps, np is num features per prev app
#   POSCashBalance:         x_pos, of size (a, pc, p, npc), where pc is num pos balances per prev app, npc is num features per balance
#   InstallmentsPayments:   x_instpay, of size (a, ip, p, nip), where ip is num payments per prev app, nip is num features per payment
#   CreditCardBalance:      x_ccbal, of size (a, c, p, nc), where c is num cc balances per prev app, nc is num features per cc balance

# Mock data dims
## Application
a = 1000
na = 125
## Bureau
b = 15
nb = 10
## BureauBalance
bb = 100
nbb = 8
## PreviousApplication
p = 20
np = 12
## POSCashBalance
pc = 50
npc = 6
## InstallmentsPayments
ip = 75
nip = 9
## CreditCardBalance
c = 80
nc = 9



class RiskDataset(Dataset):
    """Prepares data from all tables for input into the Risk model.

    Arguments are the respective tables' Dataset objects.
    If the Dataset isn't a leaf node in the Dataset tree, it comes with a
        map that designates which of its indices maps to which indices in
        its child(ren) Dataset(s).
    """
    def __init__(self, app, y, bureau, burbal, prvapp, pos, instpay, ccbal):
        super(RiskDataset, self).__init__()
        self.app = app
        self.y = y
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
        x_app = self.app[ix] # should have size (1, na)

        # get BureauDataset instance
        b_ix = self.app.bureau_map[ix]
        x_bureau = self.bureau[b_ix]

        # get BureauBalanceDataset instance
        burbal_indices = self.bureau.burbal_map[b_ix]
        x_burbal = []
        for bb_ix in burbal_indices:
            x_burbal.append(self.burbal[bb_ix].unsqueeze(1)) # TODO confirm size of each tensor in x_burbal is (bb, 1, nbb)
            print(x_burbal[-1].size())
        x_burbal = torch.cat(x_burbal, dim=1).unsqueeze(0) # should have size (1, bb, b, nbb)

        # get PreviousApplicationDataset instance
        p_ix = self.app.prvapp_map[ix]
        x_prvapp = self.prvapp[p_ix]

        # get POSCashBalanceDataset instance
        pos_indices = self.prvapp.pos_map[p_ix]
        x_pos = []
        for pc_ix in pos_indices:
            x_pos.append(self.pos[pc_ix].unsqueeze(1)) # TODO confirm size of each tensor in x_pos is (pc, 1, npc)
        x_pos = torch.cat(x_pos, dim=1).unsqueeze(0)

        # get InstallmentsPaymentsDataset instance
        instpay_indices = self.prvapp.instpay_map[p_ix]
        x_instpay = []
        for ip_ix in instpay_indices:
            x_instpay.append(self.instpay[ip_ix].unsqueeze(1))
        x_instpay = torch.cat(x_instpay, dim=1).unsqueeze(0)
        
        # get CreditCardBalance instancesDataset instance
        ccbal_indices = self.prvapp.ccbal_map[p_ix]
        x_ccbal = []
        for c_ix in ccbal_indices:
            x_ccbal.append(self.ccbal[c_ix].unsqueeze(1))
        x_ccbal = torch.cat(x_ccbal, dim=1).unsqueeze(0)

        return (x_app, x_bureau, x_prvapp, x_burbal, x_pos, x_instpay, x_ccbal), self.y[ix]

    def __len__(self):
        """Obligatory Dataset __len__ magic method.

        In this case, it's ApplicationDataset.__len__.
        """
        return len(self.app)


class ApplicationDataset(Dataset):
    """Prepares data from the application{train/test}.csv table."""
    def __init__(self, filepath, bureau_map, prvapp_map):
        super(ApplicationDataset, self).__init__()
        self.data = torch.arange(a * na, dtype=torch.float).view(a, na)
        # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
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
        self.data = torch.arange(b * a * nb, dtype=torch.float).view(b, a, nb)
        # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
        self.burbal_map = burbal_map # maps each index in the `data` iterable to a collection of indices in the BureauBalanceDataset

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)


class BureauBalanceDataset(Dataset):
    """Prepares data from the bureau_balance.csv table"""
    def __init__(self, filepath):
        super(BureauBalanceDataset, self).__init__()
        self.data = torch.arange(bb * (b * a) * nbb, dtype=torch.float).view(bb, b * a, nbb)
        # (bb, b * a, nbb) # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
        # TODO should the above be 4d (1000, 100, 15, 8) or 3d (100, 15 * 1000, 8)?

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)


class PreviousApplicationDataset(Dataset):
    """Prepares data from the previous_application.csv table"""
    def __init__(self, filepath, pos_map, instpay_map, ccbal_map):
        super(PreviousApplicationDataset, self).__init__()
        self.data = torch.arange(p * a * np, dtype=torch.float).view(p, a, np)
        # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 
        self.pos_map = pos_map # maps each index in the `data` iterable to a collection of indices in the POSCashBalanceDataset
        self.instpay_map = instpay_map # maps each index in the `data` iterable to a collection of indices in the InstallmentsPaymentsDataset
        self.ccbal_map = ccbal_map # maps each index in the `data` iterable to a collection of indices in the CreditCardBalanceDataset

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)


class POSCashBalanceDataset(Dataset):
    """Prepares data from the POS_CASH_balance.csv table"""
    def __init__(self, filepath):
        super(POSCashBalanceDataset, self).__init__()
        self.data = torch.arange(pc * (p * a) * npc, dtype=torch.float).view(pc, p * a, npc)
        # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)


class InstallmentsPaymentsDataset(Dataset):
    """Prepares data from the installments_payments.csv table"""
    def __init__(self, filepath):
        super(InstallmentsPaymentsDataset, self).__init__()
        self.data = torch.arange(ip * (p * a) * nip, dtype=torch.float).view(ip, p * a, nip)
        # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)


class CreditCardBalanceDataset(Dataset):
    """Prepares data from the credit_card_balance.csv table"""
    def __init__(self, filepath):
        super(CreditCardBalanceDataset, self).__init__()
        self.data = torch.arange(c * (p * a) * nc, dtype=torch.float).view(c, p * a, nc)
        # TODO change the above to something like `np.load(filepath, mmap_mode='r')` when processed data lands 

    def __getitem__(self, ix):
        return self.data[:, ix, :]

    def __len__(self):
        return self.data.size(1)
