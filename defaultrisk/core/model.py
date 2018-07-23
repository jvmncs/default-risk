from .layers import PooledLSTM
import torch
import torch.nn as nn

# TODO line up dimensions in Risk's forward hook
# TODO consider unsupervised pretraining for each module
# TODO figure out per-module masking

class Risk(nn.Module):
    """
    The top-level risk model.
    Container for all other modules defined below.
    """
    def __init__(self, app_in, bureau_in, prvapp_in,        # input features
                burbal_in, pos_in, instpay_in, ccbal_in,    # input features
                bureau_out, prvapp_out, burbal_out,         # intermediate representations
                pos_out, instpay_out, ccbal_out,            # intermediate representations
                hidden=256):
        super(Risk, self).__init__()

        # Modules
        self.application = Application(
                        app_in,
                        bureau_out, prvapp_out,
                        hidden)
        self.bureau = Bureau(
                        bureau_in,
                        burbal_out,
                        bureau_out)
        self.prvapp = PreviousApplication(
                        prvapp_in,
                        pos_out, instpay_out, ccbal_out,
                        prvapp_out)
        self.burbal = BureauBalance(
                        burbal_in,
                        burbal_out)
        self.pos = POSCashBalance(
                        pos_in,
                        pos_out)
        self.instpay = InstallmentsPayments(
                        instpay_in,
                        instpay_out)
        self.ccbal = CreditCardBalance(
                        ccbal_in,
                        ccbal_out)
        
    def forward(self, x_app, x_bureau, x_prvapp,
                x_burbal, x_pos, x_instpay, x_ccbal):
        # Previous Credit Home modules
        pos = self.pos(x_pos)
        instpay = self.instpay(x_instpay)
        ccbal = self.ccbal(x_ccbal)
        x = torch.cat((prvapp_in, pos, instpay, ccbal), dim=-1)
        prvapp = self.prvapp(x)

        # Bureau modules
        burbal = self.burbal(x_burbal)
        x = torch.cat((x_bureau, burbal), dim=-1)
        bureau = self.bureau(x)
        # Application module
        x = torch.cat((x_app, bureau, prvapp), dim=-1)
        return self.application(x)


class Application(nn.Module):
    """A module learning from the high-level representation of each
    credit application.  Maps applications and any other learned
    representations to a probability representing default risk.

    Table: application_{train|test}.csv
    Upstream modules: Bureau, PreviousApplication
    Downstream modules: None (this is the last classification module)
    """
    def __init__(self, app_in, bureau_out, prvapp_out,
                hidden=256):
        super(Application, self).__init__()
        self.app = nn.Sequential(
            nn.Linear(app_in, hidden),
            nn.Linear(hidden, hidden//2))
        self.classifier = nn.Linear(
                        hidden//2 + bureau_out + prvapp_out, 1)

    def forward(self, x_app, x_bureau, x_prvapp):
        x = self.app(x_app)
        x = torch.cat((x, x_bureau, x_prvapp), dim=-1)
        return self.classifier(x)


class Bureau(PooledLSTM):
    """A module learning from a sequence of past credits reported to
    the Credit Bureau from other financial organizations.
    
    Table: bureau.csv
    Upstream modules: BureauBalance
    Downstream modules: Application
    """
    def __init__(self, bureau_in, burbal_out,   # inputs
                bureau_out,                     # outputs
                dropout=0, bidirectional=False):
        super(Bureau, self).__init__(
            bureau_in + burbal_out,
            bureau_out,
            dropout, bidirectional)


class PreviousApplication(PooledLSTM):
    """A module learning from a sequence of previous applications for
    Home Credit loans.

    Table: previous_application.csv
    Upstream modules: POSCashBalance, CreditCardBalance,
        InstallmentsPayments
    Downstream modules: Application
    """
    def __init__(self, prvapp_in, pos_out, instpay_out, ccbal_out,
                prvapp_out,
                dropout=0, bidirectional=False):
        super(PreviousApplication, self).__init__(
            prvapp_in + pos_out + instpay_out + ccbal_out,
            prvapp_out,
            dropout=0, bidirectional=False)


class BureauBalance(PooledLSTM):
    """A module learning from monthly balances of past credits in the
    Credit Bureau.
    
    Table: bureau_balance.csv
    Upstream modules: None
    Downstream modules: Bureau
    """
    def __init__(self, burbal_in,
                burbal_out,
                dropout=0, bidirectional=False):
        super(BureauBalance, self).__init__(
            burbal_in,
            burbal_out,
            dropout=0, bidirectional=False)

        
class POSCashBalance(PooledLSTM):
    """A module learning from monthly balances of previous point of
    sales and cash loans with Home Credit.

    Table: POS_CASH_balance.csv
    Upstream modules: None
    Downstream modules: PreviousApplication
    """
    def __init__(self, pos_in,
                pos_out,
                dropout=0, bidirectional=False):
        super(POSCashBalance, self).__init__(
            pos_in,
            pos_out,
            dropout=0, bidirectional=False)


class InstallmentsPayments(PooledLSTM):
    """A module learning from repayment history for previously
    disbursed credits in Home Credit.

    Table: installments_payments.csv
    Upstream modules: None
    Downstream modules: PreviousApplication
    """
    def __init__(self, instpay_in,
                instpay_out,
                dropout=0, bidirectional=False):
        super(InstallmentsPayments, self).__init__(
            instpay_in,
            instpay_out,
            dropout=0, bidirectional=False)


class CreditCardBalance(PooledLSTM):
    """A module learning from mothly balanced of previous credit
    cards with Home Credit.

    Table: credit_card_balance.csv
    Upstream modules: None
    Downstream modules: PreviousApplication
    """
    def __init__(self, ccbal_in,
                ccbal_out,
                dropout=0, bidirectional=False):
        super(CreditCardBalance, self).__init__(
            ccbal_in,
            ccbal_out,
            dropout=0, bidirectional=False)
