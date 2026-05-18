import numpy as np
from scipy.io import loadmat
import torch
import random
from utils import generate_inputPNT
import deepxde as dde
from pyDOE import lhs

def generate_unique_numbers(max_num, n):
    # random.seed(1234)  # 设置随机数种子，确保每次生成的随机数序列相同
    unique_numbers = random.sample(range(0, max_num), n)  # 修改范围根据需求调整
    return unique_numbers
def generate_unique_numbers_bg(dataptnum, dense_n,tr_n):
    xrange=[-1,1]
    densrange = [-0.2,0.2]
    densptmindiv =abs((min(densrange)-min(xrange)))/abs((max(xrange)-min(xrange)))
    denspymaxdiv =abs(max(densrange)-min(xrange))/abs((max(xrange)-min(xrange)))
    dsptmin =int(dataptnum*densptmindiv)
    dsptmax =int(dataptnum*denspymaxdiv)
    ds_unique_numbers = random.sample(range(dsptmin,dsptmax),dense_n)
    l_range = range(0,dsptmin)
    r_range = range(dsptmax,dataptnum)
    tr_range = list(l_range) + list(r_range)
    tr_unique_numbers = random.sample(tr_range,tr_n-dense_n)
    #print(tr_unique_numbers)
    # random.seed(1234)  # 设置随机数种子，确保每次生成的随机数序列相同
    unique_numbers = ds_unique_numbers+tr_unique_numbers
    return unique_numbers
def load_bgdata():
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    data_x_t, data_u = X,y
    x_values = data_x_t[:, 0]  # 获取第一列，代表 x 的值

    # 获取排序索引
    sorted_indices = np.argsort(x_values)

    # icbcinput/label
    data_x_t = data_x_t[sorted_indices]
    data_u = data_u[sorted_indices]
    idxr = data_x_t[:, 0] == 1
    idxl = data_x_t[:, 0] == -1
    idxbc = idxl|idxr
    idxic = data_x_t[:,1]==0
    print(data_x_t[:,1])
    print(idxbc,idxic)
    ic_num = 100
    bc_num =100

    ic_input_all = data_x_t[idxic]
    ic_label_all = data_u[idxic]
    ic_input_all = torch.from_numpy(ic_input_all).float()
    ic_label_all = torch.from_numpy(ic_label_all).float()
    lhsic=lhs(1,ic_num)
    lhsic_idx = lhsic*ic_input_all.shape[0]
    ic_input = ic_input_all[lhsic_idx]
    ic_label = ic_label_all[lhsic_idx]
    bc_input_all = data_x_t[idxbc]
    bc_label_all = data_u[idxbc]
    bc_input_all = torch.from_numpy(bc_input_all).float()
    bc_label_all = torch.from_numpy(bc_label_all).float()
    lhsbc = lhs(1,bc_num)
    lhsbc_idx = lhsbc*bc_input_all.shape[0]
    bc_input = bc_input_all[lhsbc_idx]
    bc_label = bc_label_all[lhsbc_idx]
    #print(data_x_t) #X(25600,2),(X,T),X[-1,1]
    input_data_num = 2
    out_data_num = 1
    dense_point_num = 150
    train_point_num = 500
    train_input_point_list = torch.ones((train_point_num, input_data_num))
    train_label_point_list = torch.ones((train_point_num, out_data_num))
    test_point_num = 1000
    test_input_point_list = torch.ones((test_point_num, input_data_num))
    test_label_point_list = torch.ones((test_point_num, out_data_num))

    unique_numbers = generate_unique_numbers_bg(data_x_t.shape[0],dense_point_num, train_point_num + test_point_num)

    for i in range(0, train_point_num):
        train_input_point_list[i][0] = data_x_t[unique_numbers[i]][0]
        train_input_point_list[i][1] = data_x_t[unique_numbers[i]][1]
        train_label_point_list[i][0] = data_u[unique_numbers[i]][0]
        # print(unique_numbers[i])

    for i in range(train_point_num, train_point_num + test_point_num):
        test_input_point_list[i - train_point_num][0] = data_x_t[unique_numbers[i]][0]
        test_input_point_list[i - train_point_num][1] = data_x_t[unique_numbers[i]][1]
        test_label_point_list[i - train_point_num][0] = data_u[unique_numbers[i]][0]
    count = ((train_input_point_list[:, 0] > -0.2) & (train_input_point_list[:, 0] < 0.2)).sum().item()

    # 输出结果
    print("在区间 (-0.2, 0.2) 的值的个数:",count, "初始条件个数:")
    # PDE_input1 = generate_inputPNT(n_var=2, ranges=[[-1, 1], [0, 1]], pnt_num=10000)
    # PDE_input2 = generate_inputPNT(n_var=2, ranges=[[-0.1, 0.1], [0, 1]], pnt_num=1000)
    # PDE_input = torch.cat([PDE_input1, PDE_input2], dim=0)
    PDE_input = generate_inputPNT(n_var=2, ranges=[[-1, 1], [0, 1]], pnt_num=10000)
    dataset = {}
    dataset['train_input'] = train_input_point_list
    dataset['test_input'] = test_input_point_list
    dataset['train_label'] = train_label_point_list
    dataset['test_label'] = test_label_point_list
    dataset['pde'] = PDE_input
    dataset['bc_input'] = bc_input
    dataset['bc_label'] = bc_label
    dataset['ic_input'] = ic_input
    dataset['ic_label'] = ic_label
    print("在区间 (-0.2, 0.2) 的值的个数:", count, "初始条件个数:",ic_label.shape,"边界条件个数",bc_label.shape)
    return dataset['train_input'], dataset['test_input'], PDE_input, dataset['train_label'], dataset['test_label'],dataset['bc_input'],dataset['bc_label'],dataset['ic_input'],dataset['ic_label']
def load_acdata():
    data = loadmat("./Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    data_x_t, data_u = X,y
    input_data_num = 2
    out_data_num = 1
    train_point_num = 225
    train_input_point_list = torch.ones((train_point_num, input_data_num))
    train_label_point_list = torch.ones((train_point_num, out_data_num))
    test_point_num = 1000
    test_input_point_list = torch.ones((test_point_num, input_data_num))
    test_label_point_list = torch.ones((test_point_num, out_data_num))

    unique_numbers = generate_unique_numbers(data_x_t.shape[0], train_point_num + test_point_num)

    for i in range(0, train_point_num):
        train_input_point_list[i][0] = data_x_t[unique_numbers[i]][0]
        train_input_point_list[i][1] = data_x_t[unique_numbers[i]][1]
        train_label_point_list[i][0] = data_u[unique_numbers[i]][0]
        # print(unique_numbers[i])

    for i in range(train_point_num, train_point_num + test_point_num):
        test_input_point_list[i - train_point_num][0] = data_x_t[unique_numbers[i]][0]
        test_input_point_list[i - train_point_num][1] = data_x_t[unique_numbers[i]][1]
        test_label_point_list[i - train_point_num][0] = data_u[unique_numbers[i]][0]
    PDE_input = generate_inputPNT(n_var=2, ranges=[[-1, 1], [0, 1]], pnt_num=800)
    dataset = {}
    dataset['train_input'] = train_input_point_list
    dataset['test_input'] = test_input_point_list
    dataset['train_label'] = train_label_point_list
    dataset['test_label'] = test_label_point_list
    dataset['pde'] = PDE_input
    return dataset['train_input'],dataset['test_input'],PDE_input,dataset['train_label'],dataset['test_label']
def bg_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(0, 1, num_t).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, num_x).reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    
    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    u[:,0:1] = -np.sin(np.pi*x)
    
    t_data = T_star.flatten()[:, None]
    x_data = X_star.flatten()[:, None]
    u_data = u.flatten()[:, None]
    
    t_data_f = t_data.copy()
    x_data_f = x_data.copy()
    
    if typ == 'train':
        idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t)
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]
        
        init_idx = np.random.choice(N-1, num_x-2, replace=False) + 1
        t_data = np.concatenate([t_data, np.zeros((num_x-2,1))], axis=0)
        x_data = np.concatenate([x_data, x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, u[init_idx,0:1]], axis=0)
        
        return t_data, x_data, u_data, t_data_f, x_data_f
   
    else:
        return t_data_f, x_data_f

def ac_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(0, 1, num_t).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, num_x).reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    
    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    u[:,0:1] = (x**2)*np.cos(np.pi*x)
    u[0,:] = -np.ones(T) 
    u[-1,:] = u[0,:]
    
    t_data = T_star.flatten()[:, None]
    x_data = X_star.flatten()[:, None]
    u_data = u.flatten()[:, None]
    
    t_data_f = t_data.copy()
    x_data_f = x_data.copy()
    
    if typ == 'train':
        idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t)
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]
        
        init_idx = np.random.choice(N-1, num_x-4, replace=False) + 1
        t_data = np.concatenate([t_data, np.ones((2,1)), np.zeros((num_x-4,1))], axis=0)
        x_data = np.concatenate([x_data, np.array([[-1], [1]]), x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, -np.ones((2,1)), u[init_idx,0:1]], axis=0)
        
        return t_data, x_data, u_data, t_data_f, x_data_f
   
    else:
        return t_data_f, x_data_f
