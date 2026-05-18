from wavenet.WAVEFAN import WAVEFANBLOCK
import data
import utils
import model
import os
import random
import pickle
cwd = os.getcwd()
pdir1 = os.path.dirname(cwd)
import sys
sys.path.append(pdir1)

# Train on MNIST
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import loadmat
from pde_Data_Loader import Pde_U_V_Dataset

if torch.cuda.is_available():
   print('CUDA is available. Using GPU for computation.')
   device = torch.device('cuda')
else:
   print('CUDA is not available. Using CPU for computation.')
   device = torch.device('cpu')

def gen_testdata():
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    # print(xx.shape)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    # print(X.shape)
    y = exact.flatten()[:, None]
    # print(y.shape)
    return X, y

data_x_t, data_u = gen_testdata()
# print(data_x_t)-
# print(data_x_t.shape)

data_x_t = torch.tensor(data_x_t)
data_u = torch.tensor(data_u)


dataset = {}
data_u = data_u.to(torch.float32)
data_x_t = data_x_t.to(torch.float32)
dataset['pred_input'] = data_x_t.to(device)
dataset['pred_label'] = data_u.to(device)





input_dim=2
output_dim=1
hidden_dim=50
num_layers=6
model = WAVEFANBLOCK(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)




#model.load_ckpt("../weight.checkpoint")
checkpoint = torch.load('./N6_lr1.0e-03/weight200000.checkpoint',map_location=torch.device('cpu'))
print("Load the weight file successfully!")
print(checkpoint.keys())
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

Train_loss = checkpoint['tloss']
val_loss = checkpoint['vloss']
model(dataset['pred_input'])

model(dataset['pred_input'])

pred_u = model(dataset['pred_input'])

pred_u = pred_u.view(100, 256)
true_u_js = dataset['pred_label'].view(100, 256).cpu().detach().numpy()

# 将PyTorch张量转换为NumPy数组
pred_u = pred_u.cpu().detach().numpy()
res_u = ((((pred_u - true_u_js) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(true_u_js))))) * 100

# 重新整形为101x201的形状
pred_u = np.reshape(pred_u, (100, 256))
res_u = np.reshape(res_u, (100, 256))

# 将NumPy数组保存到文件中
np.save(f'u_predBG_PIFAN.npy', pred_u)
np.save(f'u_resBG_PIFAN.npy', res_u)





true_u = dataset['pred_label'].view(100, 256)
true_u = true_u.cpu().detach().numpy()
true_u = np.reshape(true_u, (100, 256))
np.save('u_trueBG.npy', true_u)

