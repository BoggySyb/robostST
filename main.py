import configparser
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
import numpy as np
import argparse
import os
import sys
import warnings
from data.GenerateDataset import loaddataset, load_adj, loadnpz
# from tsl.data.utils import WINDOW
import datetime
import codecs

from baselines.BiaTCGNet.BiaTCGNet import Model as BiaTCGNet
from baselines.GinAR.ginar_arch import GinAR
from baselines.DSFormer.model import DSFormer

"""
执行命令
`python main.py --dataset METR-LA  --model_name DSFormer --cudaidx 0`
`python main.py --mask_ratio 0.4 --dataset METR-LA  --model_name BitGraph --cudaidx 1`
"""


torch.multiprocessing.set_sharing_strategy('file_system')
node_number=207
parser = argparse.ArgumentParser()
parser.add_argument('--cudaidx', type=int, default=-1) # gpu选择
parser.add_argument('--model_name', type=str) # model选择
parser.add_argument('--dataset', default='Elec') # dataset选择

parser.add_argument('--seq_len',default=24,type=int) # 训练序列长度
parser.add_argument('--pred_len',default=24, type=int) # 预测序列长度
parser.add_argument('--mask_ratio',type=float,default=0.1) # 遮盖长度

# Training params
parser.add_argument('--lr', type=float, default=0.001)  #0.001
parser.add_argument('--gamme', type=float, default=0.0)
parser.add_argument('--milestone', type=str, default="")
parser.add_argument("--max_norm", type=float, default=0.0)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=64)


parser.add_argument('--task', default='prediction', type=str)
parser.add_argument("--adj-threshold", type=float, default=0.1)
parser.add_argument('--val_ratio',default=0.2)
parser.add_argument('--test_ratio',default=0.2)
parser.add_argument('--column_wise',default=False)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--fc_dropout', default=0.2, type=float)
parser.add_argument('--head_dropout', default=0, type=float)
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--kernel_set', type=list, default=[2,3,6,7], help='kernel set')

##############transformer config############################
parser.add_argument('--enc_in', type=int, default=node_number, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=node_number, help='decoder input size')
parser.add_argument('--c_out', type=int, default=node_number, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--num_nodes', type=int, default=node_number, help='dimension of fcn')
parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')
#######################AGCRN##########################
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--embed_dim', default=512, type=int)
parser.add_argument('--rnn_units', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--cheb_k', default=2, type=int)
parser.add_argument('--default_graph', type=bool, default=True)

#############GTS##################################
parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')

parser.add_argument("--config_filename", type=str, default='')
#####################################################
parser.add_argument("--config", type=str, default='imputation/spin.yaml')
parser.add_argument('--output_attention', type=bool, default=False)
# Splitting/aggregation params
parser.add_argument('--val-len', type=float, default=0.2)
parser.add_argument('--test-len', type=float, default=0.2)
# parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--l2-reg', type=float, default=0.)
# parser.add_argument('--batches-epoch', type=int, default=300)
parser.add_argument('--batch-inference', type=int, default=32)
parser.add_argument('--split-batch-in', type=int, default=1)
parser.add_argument('--grad-clip-val', type=float, default=5.)
parser.add_argument('--loss-fn', type=str, default='l1_loss')
parser.add_argument('--lr-scheduler', type=str, default=None)
# parser.add_argument('--history_len',default=24,type=int) #96
parser.add_argument('--label_len', default=12,type=int) #48
parser.add_argument('--horizon',default=24,type=int)
parser.add_argument('--delay',default=0,type=int)
parser.add_argument('--stride',default=1,type=int)
parser.add_argument('--window_lag',default=1,type=int)
parser.add_argument('--horizon_lag',default=1,type=int)

# Connectivity params
# parser.add_argument("--adj-threshold", type=float, default=0.1)
args = parser.parse_args()

# gpu 选择
device = "cpu" if args.cudaidx == -1 else "cuda:" + str(args.cudaidx)
print(f"We are using {device}!!!!")

### 数据集配置
if(args.dataset=='METR-LA'):
    node_number= 207
    args.num_nodes= 207
    args.enc_in= 207
    args.dec_in= 207
    args.c_out= 207
elif(args.dataset=='PEMS'):
    node_number= 325
    args.num_nodes= 325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif(args.dataset=='ETTh1'):
    node_number= 7
    args.num_nodes= 7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif(args.dataset=='Elec'):
    node_number= 321
    args.num_nodes= 321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif(args.dataset=='BeijingAir'):
    node_number=36
    args.num_nodes= 36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36
else:
    warnings.warn(f"Wrong dataset: {args.dataset}!!!!!!!")
    sys.exit(1)


### 模型配置
model = None
if args.model_name == "BitGraph":
    args.epochs = 200
    args.batch_size = 32
    args.lr = 1e-3
    
    model = BiaTCGNet(True, True, 2, node_number, args.kernel_set,
            device, predefined_A=None,
            dropout=0.3, subgraph_size=5,
            node_dim=3,
            dilation_exponential=1,
            conv_channels=8, residual_channels=8,
            skip_channels=16, end_channels= 32,
            seq_length=args.seq_len, in_dim=1, out_len=args.pred_len, out_dim=1,
            layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device)

elif args.model_name == "GinAR":
    args.epochs = 100
    args.batch_size = 16
    args.lr = 0.006
    args.gamme = 0.5
    args.milestone = "1,15,40,70,90"
    args.max_norm = 5

    adj_mx, _ = load_adj("./data/" + args.dataset + "/adj_"  + args.dataset + ".pkl", "doubletransition")
    adj_mx = [torch.tensor(i).float() for i in adj_mx]
    
    model = GinAR(input_len=args.seq_len, num_id=node_number, out_len=args.pred_len,
                in_size = 1, emb_size=16, grap_size=8, 
                layer_num=2, dropout=0.15, adj_mx=adj_mx).to(device)

elif args.model_name == "DSFormer":
    args.epochs = 100
    args.batch_size = 16
    args.lr = 0.0001
    args.gamme = 0.5
    args.milestone = "25,50,75"
    args.seq_len = 96
    args.pred_len = 96

    model = DSFormer(Input_len=args.seq_len, out_len=args.pred_len, num_id=node_number,
                        num_layer=2, dropout=0.15, muti_head=2, num_samp=2, IF_node=True).to(device)

else:
    warnings.warn(f"Wrong model_name: {args.model_name}!!!!!!!")
    sys.exit(1)
    
### train lr
gamme, milestone = args.gamme, []
if gamme > 0:
    milestone = [int(x) for x in args.milestone.split(',')]

### train loss
criteron1 = nn.L1Loss().to(device)
criteron2 = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

### Log dir
if args.seed < 0:
    args.seed = np.random.randint(1e9)
torch.set_num_threads(1)
exp_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
exp_name = f"{args.mask_ratio}_{exp_name}_{args.seed}"
logdir = os.path.join('./log_dir', args.dataset, args.model_name, exp_name)
# save config for logging
os.makedirs(logdir, exist_ok=True)

### model path
# os.makedirs('./output_'+args.model_name+'_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task, exist_ok=True)
# state_path = './output_'+args.model_name+'_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task+'/best.pth'

os.makedirs('./pretrained_'+args.model_name+'_'+args.dataset, exist_ok=True)
state_path = './pretrained_'+args.model_name+'_'+args.dataset+'/best.pth'


def train(model, train_dataloader, val_dataloader, scaler):

    best_loss=9999999.99
    k=0
    for epoch in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            ### x shape: B,L1,N,D  ### y shape: B,L2,N,D
            x, y = x.to(device), y.to(device)
            mask = None
            if args.model_name == "BitGraph":
                x_hat = model(x, mask, k)
            elif args.model_name == "GinAR":
                x_hat = model(x)
            elif args.model_name == "DSFormer":
                x_hat = model(x.squeeze(-1)).unsqueeze(-1)

            trn_loss = 0.35 * criteron1(x_hat, y) + 0.65 * criteron2(x_hat, y)
            # trn_loss = criteron1(x_hat, y)
            optimizer.zero_grad()  # optimizer.zero_grad()
            trn_loss.backward()
            
            if args.max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm, error_if_nonfinite=True)
            
            optimizer.step()
            if i % 100 == 0:
                with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
                    fout.write(f"Epoch {epoch} batch {i}: trn-loss = {trn_loss:.3f}\n")

            if gamme > 0:
                if (i + 1) in milestone:
                    for params in optimizer.param_groups:
                        params['lr'] *= gamme
                        params["weight_decay"] *= gamme

        val_loss = evaluate(model, val_dataloader, scaler)
        print(f"Epoch {epoch}, valid-loss = {val_loss:.3f}\n")

        with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
            fout.write(f"Epoch {epoch}: val_loss = {val_loss:.3f}\n")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, state_path)


def evaluate(model, val_iter, scaler):
    model.eval()
    loss=0.0
    k=0
    with torch.no_grad():
        for i, (x,y) in enumerate(val_iter):
            x, y = x.to(device), y.to(device)
            mask = None
            if args.model_name == "BitGraph":
                x_hat = model(x, mask, k)
            elif args.model_name == "GinAR":
                x_hat = model(x)
            elif args.model_name == "DSFormer":
                x_hat = model(x.squeeze(-1)).unsqueeze(-1)

            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)

            losses = criteron1(x_hat, y)
            loss+=losses


    return loss/len(val_iter)


def run():

    print(f"Dataset = {args.dataset}, Seq_len = {args.seq_len}, Pred_len = {args.pred_len}")
    train_dataloader, val_dataloader, _, scaler = loaddataset(args.seq_len, args.pred_len, args.dataset, args.batch_size)

    ### train
    train(model, train_dataloader, val_dataloader, scaler)


if __name__ == '__main__':
    run()