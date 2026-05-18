
import os
import random
import pickle
cwd = os.getcwd()
pdir1 = os.path.dirname(cwd)
import sys
sys.path.append(pdir1)
#from fan.FAN import FAN
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
import model

if torch.cuda.is_available():
   print('CUDA is available. Using GPU for computation.')
   device = torch.device('cuda')
else:
   print('CUDA is not available. Using CPU for computation.')
   device = torch.device('cpu')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def generate_unique_numbers(max_num, n):
    # random.seed(1234)  # 设置随机数种子，确保每次生成的随机数序列相同
    unique_numbers = random.sample(range(0, max_num), n)  # 修改范围根据需求调整
    return unique_numbers


# Load training data

if __name__ == "__main__":
    data = loadmat("../cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    #print(x.shape, y.shape)#9986  10971
      # N x 2

    #print(X_star.shape, U_star.shape, P_star.shape)

    # print(t_star.shape, X_star.shape, U_star.shape, P_star.shape)   #(200, 1) (5000, 2) (5000, 2, 200) (5000, 200)
    # N = X_star.shape[0] #N = 5000
    # T = t_star.shape[0] #T = 200
    # # print(np.max(t_star), np.min(t_star))   #19.900000000000002 0.0
    # # print(np.max(X_star[:, 0]), np.min(X_star[:, 0]))   #8.0 1.0
    # # print(np.max(X_star[:, 1]), np.min(X_star[:, 1]))   #2.0 -2.0


    # ob = np.reshape(ob_x, (50, 100))
    # print(ob.shape)   #(50, 100)
    # print(ob)
    # [[1.         1.07070707 1.14141414 ... 7.85858586 7.92929293 8.        ]
    #  [1.         1.07070707 1.14141414 ... 7.85858586 7.92929293 8.        ]
    #  [1.         1.07070707 1.14141414 ... 7.85858586 7.92929293 8.        ]
    #  ...
    #  [1.         1.07070707 1.14141414 ... 7.85858586 7.92929293 8.        ]
    #  [1.         1.07070707 1.14141414 ... 7.85858586 7.92929293 8.        ]
    #  [1.         1.07070707 1.14141414 ... 7.85858586 7.92929293 8.        ]]
    # print(ob_x.shape)   #(5000, 1)

    # print(ob_y.shape)   #(5000, 1)

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    t = torch.from_numpy(t).to(device)
    input_all = torch.cat((x, y, t), dim=1).to(torch.float).to(device)


    # num_T = len(t_star)
    # # print(num_T)    #200

    #指定时间点
    aim_T = np.array([0,1,2,3,4,5,6,7])#这里对应的是取0,1,2,3,4,5,6,7s的数据
    num_T = len(aim_T)
    num_hidden = 4
    num_nodes = 128
    layer_list = [3] + num_hidden * [num_nodes] + [3]
    model = model.pinn(layer_list).to(device)

    model.to(device)
    learn_Rate = 1e-2
    #model.load_ckpt("../weight.checkpoint")
    checkpoint = torch.load('./N[3, 128, 128, 128, 128, 3]_lr1.0e-03/weight100000.checkpoint', map_location=device)
    print("Load the weight file successfully!")
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    Train_loss = checkpoint['tloss']
    val_loss = checkpoint['vloss']

    for n in range(num_T):

        # print(aim_T[n])


        # print(ob_t.shape)   #torch.Size([5000, 1])
        dataset = {}
        #
        t_n = aim_T[n]
        #t_n = torch.tensor(t_n).to(device)
        dataset['pred_input'] = input_all[input_all[:,2] == t_n]

        pred = model(dataset['pred_input'])

        # # 将PyTorch张量转换为NumPy数组
        # u_numpy = pred.cpu().detach().numpy()

        # print(pred.shape)   #torch.Size([5000, 3])
        # print(U_star[:, :, aim_T[n]].shape) #(5000, 2)
        # print(P_star[:, aim_T[n]].shape)    #(5000,)




        # 将PyTorch张量转换为NumPy数组
        pred_u = pred[:, 0].cpu().detach().numpy()
        for_res_u = u[input_all[:,2] == t_n].squeeze()
        #print(for_res_u.shape)#）2922，1）
        for_res_v = v[input_all[:,2] == t_n].squeeze()
        for_res_p = p[input_all[:,2] == t_n].squeeze()
        res_u = ((((pred_u - for_res_u) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(for_res_u))))) * 100
        pred_v = pred[:, 1].cpu().detach().numpy()
        res_v = ((((pred_v - for_res_v) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(for_res_v))))) * 100
        pred_p = pred[:, 2].cpu().detach().numpy()
        res_p = ((((pred_p - for_res_p) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(for_res_p))))) * 100

        # print(res_u.shape, res_v.shape, res_p.shape)    #(5000,) (5000,) (5000,)

        # # 重新整形为101x201的形状
        # pred_u = np.reshape(pred_u, (50, 100))
        # res_u = np.reshape(res_u, (50, 100))
        # pred_v = np.reshape(pred_v, (50, 100))
        # res_v = np.reshape(res_v, (50, 100))
        # pred_p = np.reshape(pred_p, (50, 100))
        # res_p = np.reshape(res_p, (50, 100))

        # 将NumPy数组保存到文件中
        np.save(f'u_predNS_T{aim_T[n]}_10.npy', pred_u)
        np.save(f'u_resNS_T{aim_T[n]}_10.npy', res_u)
        np.save(f'v_predNS_T{aim_T[n]}_10.npy', pred_v)
        np.save(f'v_resNS_T{aim_T[n]}_10.npy', res_v)
        np.save(f'p_predNS_T{aim_T[n]}_10.npy', pred_p)
        np.save(f'p_resNS_T{aim_T[n]}_10.npy', res_p)

        true_u = torch.from_numpy(for_res_u)
        true_u = true_u.cpu().detach().numpy()

        np.save(f'u_trueNS_T{aim_T[n]}.npy', true_u)

        true_v = torch.from_numpy(for_res_v)
        true_v = true_v.cpu().detach().numpy()

        np.save(f'v_trueNS_T{aim_T[n]}.npy', true_v)

        true_p = torch.from_numpy(for_res_p)
        true_p = true_p.cpu().detach().numpy()

        np.save(f'p_trueNS_T{aim_T[n]}.npy', true_p)







