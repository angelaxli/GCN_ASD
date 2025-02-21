#@title GCN Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv,  global_mean_pool,global_max_pool, global_add_pool, TopKPooling
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

def graph_readout(x,batch,  method):

    if method == 'mean':
        return global_mean_pool(x,batch)

    elif method == 'meanmax':
        x_mean = global_mean_pool(x,batch)
        x_max = global_max_pool(x,batch)
        return torch.cat((x_mean, x_max),1)

    elif method == 'sum':
        return global_add_pool(x,batch)

    else:
        raise ValueError('Undefined readout opertaion')


class Abstract_GNN(torch.nn.Module):
    def __init__(self, num_nodes, readout):
        super(Abstract_GNN, self).__init__()
        self.readout = readout

    def _reset_parameters(self):
            for p in self.parameters():
                #print(p)
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    def forward(self,data):

        raise NotImplementedError


class GCN(Abstract_GNN):
    def __init__(self, num_nodes, readout, **kwargs):
        super().__init__(num_nodes, readout)
        self.f1 = 64
        self.f2 = 32
        self.readout = readout


        self.conv1 = GCNConv(num_nodes, self.f1)
        self.pool1 = TopKPooling(self.f1, ratio=0.5)
        self.conv2 = GCNConv(self.f1, self.f2)
        self.pool2 = TopKPooling(self.f2, ratio=0.5)
        last_dim = 2 if readout=='meanmax' else 1
        self.fc1 = nn.Linear((self.f2 + self.f1) * last_dim,  32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.fc2 = torch.nn.Linear(32, 2)


        self._reset_parameters()


    def forward(self, data):
        x, edge_index,edge_weight, batch, pos = data.x, data.edge_index, data.edge_attr, data.batch, data.pos
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x, edge_index, edge_weight, batch, perm1, score1 = self.pool1(x, edge_index, edge_weight, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x1 = graph_readout(x, batch, self.readout)
        pos = pos[perm1]
        s1 = zip(perm1.tolist(), score1.tolist(), batch.tolist(), pos.tolist())
        s1_cnt = {}
        for p, s, b, po in s1:
          k = 0
          for i in range(len(po)):
            if po[i] == 1:
              k = i
          if k in s1_cnt.keys():
            s1_cnt[k] += s
          else:
            s1_cnt[k] = s

        s1 = sorted(s1_cnt.items(), key = lambda x: x[1])


        x = self.conv2(x, edge_index, edge_weight)
        x= F.relu(x)
        x, edge_index, edge_weight, batch, perm2, score2 = self.pool2(x, edge_index, edge_weight, batch)
        pos = pos[perm2]
        x = F.dropout(x, p=0.5, training=self.training)
        s2 = zip(perm1[perm2].tolist(), score2.tolist(), batch.tolist(), pos.tolist())
        s2_cnt = {}

        for p, s, b, po in s2:
          k = 0
          for i in range(len(po)):
            if po[i] == 1:
              k = i
          if k in s2_cnt.keys():
            s2_cnt[k] += s
          else:
            s2_cnt[k] = s
        s2 = sorted(s2_cnt.items(), key = lambda x: x[1])


        x2 = graph_readout(x, batch, self.readout)
        x = torch.cat([x1, x2], dim = 1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=-1)

        s1, _ = zip(*s1)
        s2, _ = zip(*s2)
        return x, s1[-20:], s2[-20:]

#@title Data Preparation Functions
import os.path as osp
from os import listdir
import os
import glob
import h5py

import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
import networkx.convert_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
import pandas as pd


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    func = partial(read_sigle_data, data_dir)

    import timeit

    res = []
    for i in range(len(onlyfiles)):
      res.append(func(onlyfiles[i]))


    for j in range(len(res)):

        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch)


    data, slices = split(data, batch_torch)

    return data, slices



def read_sigle_data(data_dir,filename):

    temp = dd.io.load(osp.join(data_dir, filename))
    subject_id = filename[:5]
    node_att = pd.DataFrame([])

    file_path = os.path.join(data_dir[:-3], subject_id)
    ro_file = [f for f in os.listdir(file_path) if f.endswith('.1D')]
    file_path = os.path.join(file_path, ro_file[0])
    ho_rois = pd.read_csv(file_path, sep='\t').iloc[:78, :].T
    node_att = pd.concat([node_att, ho_rois])
    node_att = torch.tensor(node_att.values)
    pcorr = np.abs(temp['corr'][()])

    num_nodes = pcorr.shape[0]
    G = nx.DiGraph(pcorr)
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    label = temp['label'][()]

    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=node_att, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)
    # return data
    return [edge_att.data.numpy(),edge_index.data.numpy(), node_att,label,num_nodes]




     # @title Train-Validation Split

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train_val_test_split(n_sub, kfold = 10, fold = 0):
    id = list(range(n_sub))


    import random
    random.seed(42)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id

#@title ABIDE Data Loader
import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp



class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):

        self.root = root
        self.name = name
        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        return

    def process(self):
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

# @title Model Training
import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler


from torch_geometric.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=50, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='..', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.7, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=1e-2, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=78, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
opt = parser.parse_args(args=[])

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
name = 'ABIDE'

load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold



################## Define Dataloader ##################################
dataset = ABIDEDataset(path,name)

dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0
tr_index,val_index,te_index = train_val_test_split(n_sub = dataset.data.y.shape[0], fold=fold)
train_dataset = dataset[tr_index]

val_dataset = dataset[list(val_index) + list(te_index)]
test_dataset = dataset[list(val_index) + list(te_index)]


train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)



############### Define Graph Deep Learning Network ##########################
model = GCN(78, "meanmax").to(device)
print(model)

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

save_model = True

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    scheduler.step()

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    loss_all = 0
    #train by batch
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, s1, s2 = model(data)
        loss_c = F.nll_loss(output, data.y)
        loss = opt.lamb0*loss_c

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    topkdict = {}
    for data in loader:
        data = data.to(device)
        outputs, s1, s2= model(data)
        for i in s2:
          i = int(i)
          if i in topkdict.keys():
            topkdict[i]+=1
          else:
            topkdict[i] = 1
        pred = outputs.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    #selected biomarkers
    print("selected:", sorted(topkdict.items(), key=lambda
                 kv:kv[1])[-5:])

    return correct / len(loader.dataset)

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, s1, s2= model(data)
        loss_c = F.nll_loss(output, data.y)
        loss = opt.lamb0*loss_c

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
best_acc = 0
indicator = 0

for epoch in range(0, num_epoch):
    if indicator > 20:
        break
    indicator += 1
    since  = time.time()
    tr_loss= train(epoch)
    tr_acc = test_acc(train_loader)
    val_acc = test_acc(val_loader)
    val_loss = test_loss(val_loader,epoch)
    time_elapsed = time.time() - since
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                       tr_acc, val_loss, val_acc))

    if val_acc > best_acc and epoch > 5:
        indicator = 0
        print("saving best model")
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        if save_model:
            torch.save(best_model_wts, os.path.join("../.pt"))

# @title Confusion Matrix
import torch
import os
import numpy as np
import argparse
import time
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


from torch_geometric.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=50, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='..', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.7, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=1e-2, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=78, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
opt = parser.parse_args(args=[])


#################### Parameter Initialization #######################
path = opt.dataroot
name = 'ABIDE'

load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold


################## Define Dataloader ##################################
dataset = ABIDEDataset(path,name)

dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0
tr_index,val_index,te_index = train_val_test_split(n_sub = dataset.data.y.shape[0], fold=fold)
train_dataset = dataset[tr_index]

val_dataset = dataset[list(val_index) + list(te_index)]
test_dataset = dataset[list(val_index) + list(te_index)]

train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

############ Confusion Matrix and Evaluatation Metrics ################
model = GCN(78, "meanmax").to(device)
model.load_state_dict(torch.load(".pt"))
model.eval()

preds = []
trues = []
correct = 0
for data in val_loader:
    data = data.to(device)
    outputs,s1,s2= model(data)
    pred = outputs.max(1)[1]
    preds.append(pred.cpu().detach().numpy())
    trues.append(data.y.cpu().detach().numpy())
    correct += pred.eq(data.y).sum().item()
preds = np.concatenate(preds,axis=0)
trues = np.concatenate(trues,axis = 0)
cm = confusion_matrix(trues,preds, labels = [0, 1])
print("Confusion matrix")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0, 1])
disp.plot()
plt.show()
print(classification_report(trues, preds))
