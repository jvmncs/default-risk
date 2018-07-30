import torch
from defaultrisk.core.dataset import RiskDataset, ApplicationDataset, BureauDataset, BureauBalanceDataset, PreviousApplicationDataset, POSCashBalanceDataset, InstallmentsPaymentsDataset, CreditCardBalanceDataset
# from defaultrisk.core.dataset import RiskDataset, ApplicationDataset, BureauDataset, BureauBalanceDataset

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

# NOTE the tests below won't work if the product of the above dims are too large!
# This seems to be a bug in how the expected tensors are made -- likely not worth fixing

bureau_map_mock = {i:i for i in range(a)}
prvapp_map_mock = {i:i for i in range(a)}
burbal_map_mock = {i:[j + i*b for j in range(b)] for i in range(a)}
pos_map_mock = {i:[j + i*p for j in range(p)] for i in range(a)}
instpay_map_mock = {i:[j + i*p for j in range(p)] for i in range(a)}
ccbal_map_mock = {i:[j + i*p for j in range(p)] for i in range(a)}

app_d = ApplicationDataset(None, bureau_map_mock, prvapp_map_mock)
y_d = torch.ones(a, dtype=torch.float)
bureau_d = BureauDataset(None, burbal_map_mock)
burbal_d = BureauBalanceDataset(None)
prvapp_d = PreviousApplicationDataset(None, pos_map_mock, instpay_map_mock, ccbal_map_mock)
pos_d = POSCashBalanceDataset(None)
instpay_d = InstallmentsPaymentsDataset(None)
ccbal_d = CreditCardBalanceDataset(None)

# Application
def test_app():
    ix = 12
    exp_app_ix = torch.arange(ix * na, (ix + 1) * na).view(1, na)
    assert (app_d[ix] == exp_app_ix).all()

# Bureau
def test_bureau():
    ix = 3
    exp_bureau_ix = torch.cat([nb * a * i + torch.arange(nb * ix, nb * (ix + 1)).view(1, -1) for i in range(b)])
    assert (bureau_d[ix] == exp_bureau_ix).all()
    exp_burbal_map_mock = list(range(b * ix, (b * (ix + 1))))
    assert all([burbal_map_mock[ix][i] == exp_burbal_map_mock[i] for i in range(len(burbal_map_mock[ix]))])

# BureauBalance
def test_burbal():
    ix = 5
    exp_burbal_j = [torch.cat([nbb * (b * a) * i + torch.arange(nbb * jx, nbb * (jx + 1)).view(1, -1) for i in range(bb)]).unsqueeze(1) for jx in burbal_map_mock[ix]]
    burbal_j = [burbal_d[jx].unsqueeze(1) for jx in burbal_map_mock[ix]]
    exp_burbal = torch.cat(exp_burbal_j, dim=1)
    burbal = torch.cat(burbal_j, dim=1)
    assert (exp_burbal == burbal).all()

# PreviousApplication
def test_prvapp():
    ix = 27
    exp_prvapp_ix = torch.cat([np * a * i + torch.arange(np * ix, np * (ix + 1)).view(1, -1) for i in range(p)])
    assert (prvapp_d[ix] == exp_prvapp_ix).all()
    exp_pos_map_mock = list(range(p * ix, (p * (ix + 1))))
    assert all([pos_map_mock[ix][i] == exp_pos_map_mock[i] for i in range(len(pos_map_mock[ix]))])
    exp_instpay_map_mock = list(range(p * ix, (p * (ix + 1))))
    assert all([instpay_map_mock[ix][i] == exp_instpay_map_mock[i] for i in range(len(instpay_map_mock[ix]))])
    exp_ccbal_map_mock = list(range(p * ix, (p * (ix + 1))))
    assert all([ccbal_map_mock[ix][i] == exp_ccbal_map_mock[i] for i in range(len(ccbal_map_mock[ix]))])

# POSCashBalance
def test_pos():
    ix = 13
    exp_pos_j = [torch.cat([npc * (p * a) * i + torch.arange(npc * jx, npc * (jx + 1)).view(1, -1) for i in range(pc)]).unsqueeze(1) for jx in pos_map_mock[ix]]
    pos_j = [pos_d[jx].unsqueeze(1) for jx in pos_map_mock[ix]]
    exp_pos = torch.cat(exp_pos_j, dim=1)
    pos = torch.cat(pos_j, dim=1)
    assert (exp_pos == pos).all()

# InstallmentsPayments
def test_instpay():
    ix = 5
    exp_instpay_j = [torch.cat([nip * (p * a) * i + torch.arange(nip * jx, nip * (jx + 1)).view(1, -1) for i in range(ip)]).unsqueeze(1) for jx in instpay_map_mock[ix]]
    instpay_j = [instpay_d[jx].unsqueeze(1) for jx in instpay_map_mock[ix]]
    exp_instpay = torch.cat(exp_instpay_j, dim=1)
    instpay = torch.cat(instpay_j, dim=1)
    assert (exp_instpay == instpay).all()

# CreditCardBalance
def test_ccbal():
    ix = 8
    exp_ccbal_j = [torch.cat([nc * (p * a) * i + torch.arange(nc * jx, nc * (jx + 1)).view(1, -1) for i in range(c)]).unsqueeze(1) for jx in instpay_map_mock[ix]]
    ccbal_j = [ccbal_d[jx].unsqueeze(1) for jx in ccbal_map_mock[ix]]
    exp_ccbal = torch.cat(exp_ccbal_j, dim=1)
    ccbal = torch.cat(ccbal_j, dim=1)
    assert (exp_ccbal == ccbal).all()

risk_d = RiskDataset(app_d, y_d, bureau_d, burbal_d, prvapp_d, pos_d, instpay_d, ccbal_d)

def test_risk():
    ix = 7
    (x_app, x_bureau, x_prvapp, x_burbal, x_pos, x_instpay, x_ccbal), y = risk_d[ix]
    assert (x_app == app_d[ix]).all()
    assert (x_bureau == bureau_d[ix]).all()
    assert (x_prvapp == prvapp_d[ix]).all()
    
    x_burbals = torch.split(x_burbal.squeeze(), 1, dim=1)
    for i in range(len(x_burbals)):
        assert (x_burbals[i].squeeze() == burbal_d[bureau_d.burbal_map[ix][i]]).all()

    x_poss = torch.split(x_pos.squeeze(), 1, dim=1)
    for i in range(len(x_poss)):
        assert (x_poss[i].squeeze() == pos_d[prvapp_d.pos_map[ix][i]]).all()

    x_instpays = torch.split(x_instpay.squeeze(), 1, dim=1)
    for i in range(len(x_instpays)):
        assert (x_instpays[i].squeeze() == instpay_d[prvapp_d.instpay_map[ix][i]]).all()

    x_ccbals = torch.split(x_ccbal.squeeze(), 1, dim=1)
    for i in range(len(x_ccbals)):
        assert (x_ccbals[i].squeeze() == ccbal_d[prvapp_d.ccbal_map[ix][i]]).all()

test_risk()
