from wavenet.WAVEFAN import WAVEFANBLOCK
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
import data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import loadmat
from pde_Data_Loader import Pde_U_V_Dataset
from scipy.spatial import cKDTree
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




if __name__ == "__main__":
    _, _, _, _, _, input, label_h, label_u, _, _ = data.load_SWEdata()


    input = torch.from_numpy(input).float()
    label_h = torch.from_numpy(label_h).float().unsqueeze(1)
    label_u = torch.from_numpy(label_u).float().unsqueeze(1)
    #print(input.shape, label_h.shape, label_u.shape)
    data_all = torch.cat((input, label_h, label_u), dim=1)
   # print('datadatadata',data_all.shape)

    # num_points = 50
    # x_range = torch.linspace(0, 1, num_points)
    # y_range = torch.linspace(0, 1, num_points)
    # X_grid, Y_grid = torch.meshgrid(x_range, y_range)

    # Flatten the grids and combine them
    #grid_points = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)

    # Select the corresponding data from data_all (with interpolation or nearest neighbor approach)
    # For performance, you might want to convert data_all to a numpy array for lookup.
    data_all_np = data_all.numpy()

    # Find nearest points


    # tree = cKDTree(data_all_np[:, :2])  # use the x and y for the tree
    # distances, indices = tree.query(grid_points.numpy())  # query for nearest points

    # Extract corresponding u, v, p values
    selected_data = data_all

    # Now selected_data contains 500*500 points for x, y, u, v, p
    x_selected = selected_data[:, 0].unsqueeze(1)
    t_selected = selected_data[:, 1].unsqueeze(1)
    u_selected = selected_data[:, 5].unsqueeze(1)
    h_selected = selected_data[:, 4].unsqueeze(1)


    pred_x_t = input
    num_hidden = 6
    num_nodes = 50
    layer_list = [4] + num_hidden * [num_nodes] + [2]
    model = model.pinn(layer_list).to(device)

    model.to(device)

    #model.load_ckpt("../weight.checkpoint")
    checkpoint = torch.load('./N[4, 50, 50, 50, 50, 50, 50, 2]_lr1.0e-03/weight50000.checkpoint', map_location=device)
    print("Load the weight file successfully!")
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    #Train_loss = checkpoint['tloss']
    #val_loss = checkpoint['vloss']
    print(pred_x_t.shape)
    pred = model(pred_x_t)

    pred_h = pred[:, 0]
    true_h_js = h_selected.cpu().detach().numpy()
    pred_u = pred[:, 1]
    true_u_js = u_selected.cpu().detach().numpy()
    true_hu_js = (h_selected*u_selected).cpu().detach().numpy()


    # 将PyTorch张量转换为NumPy数组
    pred_h = pred_h.cpu().detach().numpy()
    pred_u = pred_u.cpu().detach().numpy()
    true_h_js = true_h_js.squeeze()
    true_hu_js = true_hu_js.squeeze()

    res_h = ((((pred_h - true_h_js) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(true_h_js))))) * 100
    pred_hu = (pred_h*pred_u)
    #print(pred_hu.shape)
    res_hu = ((((pred_hu - true_hu_js) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(true_hu_js))))) * 100



    # pred_u = np.reshape(pred_u, (51, 51))
    # res_u = np.reshape(res_u, (51, 51))
    # pred_v = np.reshape(pred_v, (51, 51))
    # res_v = np.reshape(res_v, (51, 51))
    # pred_p = np.reshape(pred_p, (51, 51))
    # res_p = np.reshape(res_p, (51, 51))

    # 将NumPy数组保存到文件中
    pred_h = pred_h.reshape(121,61)
    pred_hu = pred_hu.reshape(121,61)
    res_hu = res_hu.reshape(121,61)
    res_h = res_h.reshape(121,61)
    print(pred_hu.shape, res_hu.shape, pred_h.shape, res_h.shape)
    np.save(f'h_pred_PINN.npy', pred_h)
    np.save(f'hu_pred_PINN.npy', pred_hu)
    np.save(f'h_res_PINN.npy', res_h)
    np.save(f'hu_res_PINN.npy', res_hu)




    true_h = h_selected
    true_u = u_selected
    true_h = true_h.cpu().detach().numpy()
    #true_u = np.reshape(true_u, (51, 51))
    np.save('h_true.npy', true_h)

    true_h = h_selected
    true_hu = (true_h*true_u).cpu().detach().numpy()
    #true_v = np.reshape(true_v, (51, 51))
    np.save('hu_true.npy', true_hu)

    # true_p = p_selected.view(51, 51)
    # true_p = true_p.cpu().detach().numpy()
    # true_p = np.reshape(true_p, (51, 51))
    # np.save('p_truecav.npy', true_p)

