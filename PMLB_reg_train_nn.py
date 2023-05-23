from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pickle
from tqdm import tqdm
import os
from os.path import exists

from pmlb import fetch_data, classification_dataset_names,regression_dataset_names

import torch
import torch.nn as nn
import MFCF.gain_fucntions as gf
import MFCF.MFCF as MFCF
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from itertools import permutations,combinations
from skorch import NeuralNetRegressor
import sparselinear.sparselinear as sl





class HNN(nn.Module):

    def __init__(self, l1, l2, l3, l4, c1, c2, c3):
        super(HNN, self).__init__()
        self.sl1 = sl.SparseLinear(l1, l2, connectivity=torch.tensor([c1[1], c1[0]],dtype=torch.int64))
        self.fc1 = nn.Linear(l1, 1)

        self.sl2 = sl.SparseLinear(l2, l3, connectivity=torch.tensor([c2[1], c2[0]],dtype=torch.int64))
        self.fc2 = nn.Linear(l2, 1)

        self.c3=c3

        if len(self.c3[0]) != 0:
            self.sl3 = sl.SparseLinear(l3, l4, connectivity=torch.tensor([c3[1], c3[0]],dtype=torch.int64))
            self.fc3 = nn.Linear(l3, 1)

            self.fc4 = nn.Linear(l4, 1)

            self.read_out = nn.Linear(4, 1)
        else:
            self.sl3 = None
            self.fc3 = nn.Linear(l3, 1)

            self.fc4 = None

            self.read_out = nn.Linear(3, 1)

    def forward(self, x):
        x_f1 = F.relu(self.fc1(x))
        x_s1 = F.relu(self.sl1(x))  # shape l2

        x_f2 = F.relu(self.fc2(x_s1))
        x_s2 = F.relu(self.sl2(x_s1))

        if len(self.c3[0]) != 0:
            x_f3 = F.relu(self.fc3(x_s2))
            x_s3 = F.relu(self.sl3(x_s2))

            x_f4 = F.relu(self.fc4(x_s3))

            x = self.read_out(torch.cat([x_f1, x_f2, x_f3, x_f4], 1))
        else:
            x_f3 = F.relu(self.fc3(x_s2))

            x = self.read_out(torch.cat([x_f1, x_f2, x_f3], 1))
        return x

class HNN_adv(nn.Module):

    def __init__(self, l1, l2, l3, l4, c1, c2, c3):
        super(HNN_adv, self).__init__()
        self.sl1 = sl.SparseLinear(l1, l2, connectivity=torch.tensor([c1[1], c1[0]],dtype=torch.int64))
        self.fc1 = nn.Linear(l1, 1)

        self.sl2 = sl.SparseLinear(l2, l3, connectivity=torch.tensor([c2[1], c2[0]],dtype=torch.int64))
        self.fc2 = nn.Linear(l2, 1)

        self.c3=c3

        if len(self.c3[0]) != 0:
            self.sl3 = sl.SparseLinear(l3, l4, connectivity=torch.tensor([c3[1], c3[0]],dtype=torch.int64))
            self.fc3 = nn.Linear(l3, 1)

            self.fc4 = nn.Linear(l4, 1)

            self.read_out_1 = nn.Linear(4, 6)
            self.read_out_2 = nn.Linear(6, 1)
        else:
            self.sl3 = None
            self.fc3 = nn.Linear(l3, 1)

            self.fc4 = None

            self.read_out_1 = nn.Linear(3, 3)
            self.read_out_2= nn.Linear(3, 1)

    def forward(self, x):
        x_f1 = F.relu(self.fc1(x))
        x_s1 = F.relu(self.sl1(x))  # shape l2

        x_f2 = F.relu(self.fc2(x_s1))
        x_s2 = F.relu(self.sl2(x_s1))

        if len(self.c3[0]) != 0:
            x_f3 = F.relu(self.fc3(x_s2))
            x_s3 = F.relu(self.sl3(x_s2))

            x_f4 = F.relu(self.fc4(x_s3))

            x = F.relu(self.read_out_1(torch.cat([x_f1, x_f2, x_f3, x_f4], 1)))
            x= self.read_out_2(x)
        else:
            x_f3 = F.relu(self.fc3(x_s2))

            x = F.relu(self.read_out_1(torch.cat([x_f1, x_f2, x_f3], 1)))
            x=self.read_out_2(x)

        return x

class MLP(nn.Module):

    def __init__(self, l1, l2, l3, l4, c1, c2, c3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(l1, l2)

        self.fc2 = nn.Linear(l2, l3)
        self.c3=c3
        if len(self.c3[0]) != 0:

            self.fc3 = nn.Linear(l3, l4)
            self.fc4 = nn.Linear(l4, 1)

        else:
            self.fc3 = nn.Linear(l3, 1)
            self.fc4 = None

        self.read_out = nn.Linear(1, 1)

    def forward(self, x):
        x=x.float()
        x_f1 = F.relu(self.fc1(x))

        x_f2 = F.relu(self.fc2(x_f1))

        x_f3 = F.relu(self.fc3(x_f2))

        if len(self.c3[0]) != 0:
            x_f4 = F.relu(self.fc4(x_f3))
            x = self.read_out(x_f4)
        else:
            x = self.read_out(x_f3)

        return x

class MLPres(nn.Module):

    def __init__(self, l1, l2, l3, l4, c1, c2, c3):
        super(MLPres, self).__init__()
        self.sl1 = nn.Linear(l1, l2)
        self.fc1 = nn.Linear(l1, 1)

        self.sl2 = nn.Linear(l2, l3)
        self.fc2 = nn.Linear(l2, 1)

        self.c3 = c3

        if len(self.c3[0]) != 0:
            self.sl3 = nn.Linear(l3, l4)
            self.fc3 = nn.Linear(l3, 1)

            self.fc4 = nn.Linear(l4, 1)

            self.read_out = nn.Linear(4, 1)
        else:
            self.sl3 = None
            self.fc3 = nn.Linear(l3, 1)

            self.fc4 = None

            self.read_out = nn.Linear(3, 1)

    def forward(self, x):
        x_f1 = F.relu(self.fc1(x))
        x_s1 = F.relu(self.sl1(x))  # shape l2

        x_f2 = F.relu(self.fc2(x_s1))
        x_s2 = F.relu(self.sl2(x_s1))

        if len(self.c3[0]) != 0:
            x_f3 = F.relu(self.fc3(x_s2))
            x_s3 = F.relu(self.sl3(x_s2))

            x_f4 = F.relu(self.fc4(x_s3))

            x = self.read_out(torch.cat([x_f1, x_f2, x_f3, x_f4], 1))
        else:
            x_f3 = F.relu(self.fc3(x_s2))

            x = self.read_out(torch.cat([x_f1, x_f2, x_f3], 1))

        return x

class HNN_skip(nn.Module):

    def __init__(self, l1, l2, l3, l4, c1, c2, c3, d2, d3):
        super(HNN_skip, self).__init__()
        self.d2 = d2
        self.d3 = d3
        self.c3 = c3
        readout_in = 1

        self.sl1 = sl.SparseLinear(l1, l2, connectivity=torch.tensor([c1[1], c1[0]],dtype=torch.int64))

        self.sl2 = sl.SparseLinear(l2, l3, connectivity=torch.tensor([c2[1], c2[0]],dtype=torch.int64))
        if len(d2) != 0:
            self.skip2 = sl.SparseLinear(l2, 1, connectivity=torch.tensor([[0] * len(d2), d2],dtype=torch.int64))
            readout_in += 1

        if len(self.c3[0]) != 0:

            self.sl3 = sl.SparseLinear(l3, l4, connectivity=torch.tensor([c3[1], c3[0]],dtype=torch.int64))
            if len(d3) != 0:
                self.skip3 = sl.SparseLinear(l3, 1, connectivity=torch.tensor([[0] * len(d3), d3],dtype=torch.int64))
                readout_in += 1

            self.fc4 = nn.Linear(l4, 1)

            self.read_out = nn.Linear(readout_in, 1)

        else:
            self.sl3=None
            self.fc4=None
            self.fc3 = nn.Linear(l3, 1)
            self.read_out = nn.Linear(readout_in, 1)

    def forward(self, x):
        x_sk_ls = torch.tensor([])
        x_s1 = F.relu(self.sl1(x))  # shape l2

        x_s2 = F.relu(self.sl2(x_s1))
        if len(self.d2) != 0:
            x_sk2 = F.relu(self.skip2(x_s1))
            x_sk_ls = torch.cat([x_sk_ls, x_sk2],1)
        if len(self.c3[0]) != 0:
            x_s3 = F.relu(self.sl3(x_s2))
            if len(self.d3) != 0:
                x_sk3 = F.relu(self.skip3(x_s2))
                x_sk_ls = torch.cat([x_sk_ls, x_sk3],1)
            x_f4 = F.relu(self.fc4(x_s3))
            x = self.read_out(torch.cat([x_sk_ls, x_f4], 1))
        else:
            x_f3=F.relu(self.fc3(x_s2))
            x = self.read_out(torch.cat([x_sk_ls, x_f3], 1))
        # x = self.read_out(x_f4)
        return x

def train_model(train_X, test_X, train_y, test_y, dataset_name, model_name, model, params):
    path = f'Results/PMLB/prediction/regression_raw/{dataset_name}'
    file_path = path + '/' + f'{model_name}.pkl'
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        pass

    if not os.path.exists(file_path):
        CV = GridSearchCV(model, params,n_jobs=1)
        CV.fit(train_X, train_y.reshape(-1,1))

        pred_y = CV.best_estimator_.predict(test_X)
        with open(file_path, 'wb') as f:
            pickle.dump([test_y, pred_y], f)

def model_MLP(clique_1, clique_2, clique_3, clique_4,connection_1,connection_2,connection_3
              ,disconnection_2,disconnection_3):

    hyper_params = {'max_epochs': [50,100,300,500],
                    'lr':[0.001,0.005,0.01,0.05,0.1],
                    'optimizer__weight_decay':[0,0.05,0.1,0.2]
                    }
    mlp=MLP(len(clique_1), len(clique_2), len(clique_3), len(clique_4),connection_1,connection_2,connection_3)
    est = NeuralNetRegressor(mlp
                             , optimizer=optim.Adam
                             , optimizer__weight_decay=0.05
                             , criterion=nn.MSELoss
                             , max_epochs=100
                             , lr=0.001
                             , verbose=0)

    return est, hyper_params
def model_MLPres(clique_1, clique_2, clique_3, clique_4,connection_1,connection_2,connection_3
              ,disconnection_2,disconnection_3):

    hyper_params = {'max_epochs': [50,100,300,500],
                    'lr':[0.001,0.005,0.01,0.05,0.1],
                    'optimizer__weight_decay':[0,0.05,0.1,0.2]
                    }
    mlpres=MLPres(len(clique_1), len(clique_2), len(clique_3), len(clique_4),connection_1,connection_2,connection_3)
    est = NeuralNetRegressor(mlpres
                             , optimizer=optim.Adam
                             , optimizer__weight_decay=0.05
                             , criterion=nn.MSELoss
                             , max_epochs=100
                             , lr=0.001
                             , verbose=0)

    return est, hyper_params

def model_HNN(clique_1, clique_2, clique_3, clique_4,connection_1,connection_2,connection_3,
              disconnection_2,disconnection_3):

    hyper_params = {'max_epochs': [50,100,300,500],
                    'lr':[0.001,0.005,0.01,0.05,0.1],
                    'optimizer__weight_decay':[0,0.05,0.1,0.2]
                    }
    hnn=HNN(len(clique_1), len(clique_2), len(clique_3), len(clique_4),connection_1,connection_2,connection_3)
    est = NeuralNetRegressor(hnn
                             , optimizer=optim.Adam
                             , optimizer__weight_decay=0.05
                             , criterion=nn.MSELoss
                             , max_epochs=100
                             , lr=0.001
                             , verbose=0)

    return est, hyper_params

def model_HNN_adv(clique_1, clique_2, clique_3, clique_4,connection_1,connection_2,connection_3,
              disconnection_2,disconnection_3):

    hyper_params = {'max_epochs': [50,100,300,500],
                    'lr':[0.001,0.005,0.01,0.05,0.1],
                    'optimizer__weight_decay':[0,0.05,0.1,0.2]
                    }
    hnn_adv=HNN_adv(len(clique_1), len(clique_2), len(clique_3), len(clique_4),connection_1,connection_2,connection_3)
    est = NeuralNetRegressor(hnn_adv
                             , optimizer=optim.Adam
                             , optimizer__weight_decay=0.05
                             , criterion=nn.MSELoss
                             , max_epochs=100
                             , lr=0.001
                             , verbose=0)

    return est, hyper_params

def model_HNN_skip(clique_1, clique_2, clique_3, clique_4,connection_1,connection_2,connection_3,
              disconnection_2,disconnection_3):

    hyper_params = {'max_epochs': [50,100,300,500],
                    'lr':[0.001,0.005,0.01,0.05,0.1],
                    'optimizer__weight_decay':[0,0.05,0.1,0.2]
                    }
    hnn_skip=HNN_skip(len(clique_1), len(clique_2), len(clique_3), len(clique_4),connection_1,connection_2,connection_3,
                 disconnection_2,disconnection_3)
    est = NeuralNetRegressor(hnn_skip
                             , optimizer=optim.Adam
                             , optimizer__weight_decay=0.05
                             , criterion=nn.MSELoss
                             , max_epochs=100
                             , lr=0.001
                             , verbose=0)

    return est, hyper_params

def MFCF_J(X,max_clique_size=4):
    '''
    sparse J
    '''
    C = np.cov(X, rowvar=False)

    ctl = MFCF.mfcf_control()
    ctl['threshold'] = 0.1
    ctl['drop_sep'] = 0
    ctl['min_clique_size'] =2
    ctl['max_clique_size'] = 4
    gain_function = gf.sumsquares_gen

    cliques, separators, peo, gt = MFCF.mfcf(C, ctl, gain_function)

    J = MFCF.logo(C, cliques, separators)

    return J

def separating_cliques(G):
    clique_1 = []
    clique_2 = []
    clique_3 = []
    clique_4 = []
    for clique in nx.enumerate_all_cliques(G):
        clique = set(clique)
        if len(clique) == 1:
            clique_1.append(clique)
        elif len(clique) == 2:
            clique_2.append(clique)
        elif len(clique) == 3:
            clique_3.append(clique)
        elif len(clique) == 4:
            clique_4.append(clique)
    return clique_1,clique_2,clique_3,clique_4

def get_connection(clique_last, clique_next):
    connection_list = [[], []]
    component_mapping = {i: x for i, x in enumerate(clique_last)}
    for i, clique in enumerate(clique_next):
        component = [set(x) for x in combinations(clique, len(clique) - 1)]
        index_next = i
        index_last = [list(component_mapping.keys())[list(component_mapping.values()).index(x)] for x in component]
        for j in index_last:
            connection_list[0].append(j)
            connection_list[1].append(i)

    return connection_list

def get_disconnection(c_last, c_next):
    all_simplex = set(c_last[1])
    linked_simplex = set(c_next[0])
    disconnected_simplex = list(all_simplex - linked_simplex)

    return disconnected_simplex


if __name__ == "__main__":
    #1191_BNG_pbc
    model_dict={'MLPres':model_MLPres}
    n=len(model_dict)*len(regression_dataset_names)
    pbar = tqdm(total=n, desc='Back Test Progress', )

    for regression_dataset in regression_dataset_names:

        #print(regression_dataset)
        X, y = fetch_data(regression_dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)

        try:
            J = MFCF_J(train_X)
            err=False
        except:
            err=True

        if not err:
            G = nx.from_numpy_array(J)

            clique_1, clique_2, clique_3, clique_4 = separating_cliques(G)

            connection_1 = get_connection(clique_1, clique_2)
            connection_2 = get_connection(clique_2, clique_3)
            connection_3 = get_connection(clique_3, clique_4)

            disconnection_2 = get_disconnection(connection_1, connection_2)
            disconnection_3 = get_disconnection(connection_2, connection_3)

            if len(connection_2[0])!=0 and len(connection_3[0])!=0:
                for model_name,func in model_dict.items():
                    model,params=func(clique_1, clique_2, clique_3, clique_4,connection_1,connection_2,connection_3,
                                      disconnection_2,disconnection_3)
                    try:
                        train_model(torch.from_numpy(train_X).float(), torch.from_numpy(test_X).float(),
                                    torch.from_numpy(train_y).float(), torch.from_numpy(test_y).float(),
                                    regression_dataset,model_name, model, params)
                    except Exception as err:
                        print(regression_dataset)
                        print(err)


        pbar.update(1)





