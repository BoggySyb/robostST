import os
import random
from typing import Optional, Sequence, List
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import pickle
from data.adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix

np.set_printoptions(threshold=np.inf)

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class TSDataset(Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])

        return data,label


def load_pkl(pickle_file: str) -> object:
    """Load pickle data.
    Args:
        pickle_file (str): file path
    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def load_adj(file_path: str, adj_type: str):
    """load adjacency matrix.
    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type
    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        # PeMS04
        adj_mx = load_pkl(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx


def get_0_1_array(array, rate=0.2):
    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    np.random.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array


def synthetic_data(dataset):

    if(dataset=='METR-LA'):
        path = os.path.join('./data/METR-LA/', 'metr_la.h5')
        data = pd.read_hdf(path)
        data = np.array(data)
        data = data[:, :, None]


    elif(dataset=='PEMS'):
        path = os.path.join('./data/pems_bay/', 'pems_bay.h5')
        data = pd.read_hdf(path)
        data = np.array(data)
        data = data[:, :, None]


    elif(dataset=='ETTh1'):
        df_raw = pd.read_csv('./data/ETTh1/ETTh1.csv')
        data=np.array(df_raw)
        data=data[::,1:]
        data = data[:, :, None].astype('float32')

    elif (dataset == 'Elec'):
        data_list = []
        with open('./data/Electricity/electricity.txt', 'r') as f:
            reader = f.readlines()
            for row in reader:
                data_list.append(row.split(','))

        data = np.array(data_list).astype('float')
        data = data[:, :, None].astype('float32')

    elif(dataset=='BeijingAir'):

        data = pd.DataFrame(pd.read_hdf('./data/air_quality/small36.h5', 'pm25'))
        data=np.array(data)
        eval_mask=~np.isnan(data)
        data[np.isnan(data)]=0.0
        data = data[:, :, None].astype('float32')


    return data


def split_data_by_ratio(x, y, val_ratio, tst_ratio):
    idx = np.arange(x.shape[0])
    idx_shuffle = idx.copy()
    
    data_len = x.shape[0]
    n_val_zeros = int(data_len * val_ratio)
    n_tst_zeros = int(data_len * tst_ratio)
    
    
    tst_x = x[idx_shuffle[-n_tst_zeros:]]
    tst_y = y[idx_shuffle[-n_tst_zeros:]]
    # np_tst_mask = np.ones(tst_x.size)
    # np_tst_mask[:int(n_tst_zeros * mask_ratio)] = 0
    # np.random.shuffle(np_tst_mask)
    # np_tst_mask = np_tst_mask.reshape(tst_x.shape).astype('int32')
    

    val_x = x[idx_shuffle[-(n_val_zeros+n_tst_zeros):-n_tst_zeros]]
    val_y = y[idx_shuffle[-(n_val_zeros+n_tst_zeros):-n_tst_zeros]]
    # np_val_mask = np.ones(val_x.size)
    # np_val_mask[:int(n_val_zeros * mask_ratio)] = 0
    # np.random.shuffle(np_val_mask)
    # np_val_mask = np_val_mask.reshape(val_x.shape).astype('int32')
    

    trn_x = x[idx_shuffle[:-(n_val_zeros+n_tst_zeros)]]
    trn_y = y[idx_shuffle[:-(n_val_zeros+n_tst_zeros)]]
    # np_trn_mask = np.ones(trn_x.size)
    # np_trn_mask = np_trn_mask.reshape(trn_x.shape).astype('int32')
    

    return trn_x,trn_y, val_x,val_y, tst_x,tst_y


def Add_Window_Horizon(data, window=3, horizon=1):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []  #horizon
    index = 0

    while index < end_index:
        X.append(data[index:index+window])
        Y.append(data[index+window:index+window+horizon])
        index = index + 1
    X = np.array(X)  #backcast B,W,N,D
    Y = np.array(Y)  #forecast B,H,N,D
    
    return X, Y


def loaddataset(history_len, pred_len, dataset, batch_size):
    data_numpy = synthetic_data(dataset)
    
    x, y = Add_Window_Horizon(data_numpy, history_len, pred_len)

    trn_x,trn_y, val_x,val_y, tst_x,tst_y = split_data_by_ratio(x, y, 0.2, 0.2)

    ### data
    np.savez('./data/' + dataset + "/data.npz",
        trn_x = trn_x,
        trn_y = trn_y,
        val_x = val_x,
        val_y = val_y,
        tst_x = tst_x,
        tst_y = tst_y)

    scaler = StandardScaler(mean=trn_x.mean(), std=trn_y.std())
    x_trn = scaler.transform(trn_x)
    y_trn = scaler.transform(trn_y)
    x_val = scaler.transform(val_x)
    y_val = scaler.transform(val_y)
    x_tst = scaler.transform(tst_x)
    y_tst = scaler.transform(tst_y)

    train_dataset = TSDataset(x_trn, y_trn)
    val_dataset = TSDataset(x_val, y_val)
    test_dataset = TSDataset(x_tst, y_tst)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)


    return train_dataloader, val_dataloader, test_dataloader, scaler


def loadnpz(dataset, batch_size):
    data_path = './data/' + dataset + "/data.npz"
    raw_data = np.load(data_path, allow_pickle=True)
    
    trn_x = raw_data["trn_x"]
    trn_y = raw_data["trn_y"]
    val_x = raw_data["val_x"]
    val_y = raw_data["val_y"]
    tst_x = raw_data["tst_x"]
    tst_y = raw_data["tst_y"]


    scaler = StandardScaler(mean=trn_x.mean(), std=trn_y.std())
    x_trn = scaler.transform(trn_x)
    y_trn = scaler.transform(trn_y)
    x_val = scaler.transform(val_x)
    y_val = scaler.transform(val_y)
    x_tst = scaler.transform(tst_x)
    y_tst = scaler.transform(tst_y)

    train_dataset = TSDataset(x_trn, y_trn)
    val_dataset = TSDataset(x_val, y_val)
    test_dataset = TSDataset(x_tst, y_tst)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)


    return train_dataloader, val_dataloader, test_dataloader, scaler

if __name__ == '__main__':
    print('')