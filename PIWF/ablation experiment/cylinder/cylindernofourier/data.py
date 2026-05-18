import numpy as np
from scipy.io import loadmat
import torch
import random
from utils import generate_inputPNT, create_dataset
import deepxde as dde
import math
import scipy.io as sio
from scipy.spatial import cKDTree
from pyDOE import lhs
def latin_hypercube_2d_uniform(n):
    lower_limits=np.arange(0,n)/n
    upper_limits=np.arange(1,n+1)/n

    points=np.random.uniform(low=lower_limits,high=upper_limits,size=[2,n]).T
    np.random.shuffle(points[:,1])
    return points
torch.pi  = math.pi


def generate_unique_numbers(max_num, n):
    # random.seed(1234)  # 设置随机数种子，确保每次生成的随机数序列相同
    unique_numbers = random.sample(range(0, max_num), n)  # 修改范围根据需求调整
    return unique_numbers


def load_newcylinderdata(datafile, num_train, num_test, num_ic):
    data = loadmat(f"{datafile}")
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
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([t,x, y, u, v, p], 1)
    data2 = data1[:, :][data1[:, 0] <= 7]
    data3 = data2[:, :][data2[:, 1] >= 1]
    data4 = data3[:, :][data3[:, 1] <= 8]
    data5 = data4[:, :][data4[:, 2] >= -2]
    data_domain = data5[:, :][data5[:, 2] <= 2]
    
    # print(data_domain.shape[0]) #4w
    # choose number of training points: num =7000

    # train points
    data0 = data_domain[data_domain[:, 0] == 0]
    idx_ic = np.random.choice(data0.shape[0], num_ic, replace=False)
    t_ic = data0[idx_ic, 0:1]
    x_ic = data0[idx_ic, 1:2]
    y_ic = data0[idx_ic, 2:3]
    u_ic = data0[idx_ic, 3:4]
    v_ic = data0[idx_ic, 4:5]
    p_ic = data0[idx_ic, 5:6]
    ic_input = np.concatenate((x_ic, y_ic, t_ic), 1)
    ic_label = np.concatenate((u_ic, v_ic, p_ic), 1)
    time_list = np.unique(data_domain[:, 0])
    print(time_list)
    data_list_train = []
    data_list_test = []
    bc_input_list = []
    bc_label_list = []
    t_span = len(time_list)
    n_test = int(num_test / t_span)
    n_train = int(num_train / t_span)
    print('test_point_number_for_each_time', n_test, 'train_point_number_for_each_time', n_train)
    for i in time_list:
        data_domain_2 = data_domain[data_domain[:, 0] == i]
        #print(data_domain_2)
        # idxr = np.round(data_domain_2[:, 1], 1) == 8
        # idxl = np.round(data_domain_2[:, 1], 2) == 0.50
        # idxu = np.round(data_domain_2[:, 2], 2) == 2
        # idxd = np.round(data_domain_2[:, 2], 1) == -2.0
        idxr = data_domain_2[:, 1] == 8
        idxl = data_domain_2[:, 1] == 1
        idxu = data_domain_2[:, 2] == 2
        idxd = data_domain_2[:, 2] == -2
        bc_input_r = data_domain_2[idxr][:, 0:3][:, [1, 2, 0]]
        bc_input_l = data_domain_2[idxl][:, 0:3][:, [1, 2, 0]]
        bc_input_u = data_domain_2[idxu][:, 0:3][:, [1, 2, 0]]
        bc_input_d = data_domain_2[idxd][:, 0:3][:, [1, 2, 0]]
        #print(bc_input_r.shape, bc_input_l.shape, bc_input_u.shape, bc_input_d.shape)
        bc_output_r = data_domain_2[idxr][:, 3:]
        bc_output_l = data_domain_2[idxl][:, 3:]
        bc_output_u = data_domain_2[idxu][:, 3:]
        bc_output_d = data_domain_2[idxd][:, 3:]
        bc_input_r = torch.from_numpy(bc_input_r).float()
        # print('aaaaa',bc_input_r.shape)
        bc_input_l = torch.from_numpy(bc_input_l).float()
        bc_input_u = torch.from_numpy(bc_input_u).float()
        bc_input_d = torch.from_numpy(bc_input_d).float()
        bc_output_r = torch.from_numpy(bc_output_r).float()
        bc_output_l = torch.from_numpy(bc_output_l).float()
        bc_output_u = torch.from_numpy(bc_output_u).float()
        bc_output_d = torch.from_numpy(bc_output_d).float()
        # print(bc_input_r.shape, bc_input_l.shape, bc_input_u.shape, bc_input_d.shape)


        #         lhsbc_r_idx = int(lhsbc_y * bc_input_r.shape[0])
        #         lhsbc_l_idx = int(lhsbc_y * bc_input_l.shape[0])
        #         lhsbc_u_idx = int(lhsbc_x * bc_input_u.shape[0])
        #         lhsbc_d_idx = int(lhsbc_x * bc_input_d.shape[0])

        # lhsbc_r_idx = lhsbc_y * bc_input_r.shape[0]
        # lhsbc_l_idx = lhsbc_y * bc_input_l.shape[0]
        # lhsbc_u_idx = lhsbc_x * bc_input_u.shape[0]
        # lhsbc_d_idx = lhsbc_x * bc_input_d.shape[0]


        lhs_input_r = bc_input_r
        lhs_input_l = bc_input_l
        lhs_input_u = bc_input_u
        lhs_input_d = bc_input_d

        # print('2222222222222222222222')
        # print(lhs_input_r.shape,lhs_input_l.shape,lhs_input_u.shape,lhs_input_d.shape)

        lhs_output_r = bc_output_r
        lhs_output_l = bc_output_l
        lhs_output_u = bc_output_u
        lhs_output_d = bc_output_d
        # print(lhs_output_d.shape)
        bc_input = torch.concatenate((lhs_input_r, lhs_input_l, lhs_input_u, lhs_input_d), axis=0)
        bc_label = torch.concatenate((lhs_output_r, lhs_output_l, lhs_output_u, lhs_output_d), axis=0)
        # print(bc_input.shape)
        bc_input_list.append(bc_input)
        bc_label_list.append(bc_label)

        x_min, x_max = data_domain_2[:, 1].min(), data_domain_2[:, 1].max()
        y_min, y_max = data_domain_2[:, 2].min(), data_domain_2[:, 2].max()
        lhs_train = latin_hypercube_2d_uniform(n_train)
        lhs_test = latin_hypercube_2d_uniform(n_test)
        x_sample_train = x_min + lhs_train[:, 0] * (x_max - x_min)
        y_sample_train = y_min + lhs_train[:, 1] * (y_max - y_min)
        x_sample_test = x_min + lhs_test[:, 0] * (x_max - x_min)
        y_sample_test = y_min + lhs_test[:, 1] * (y_max - y_min)
        tree = cKDTree(data_domain_2[:, 1:3])
        _, idx_train = tree.query(np.column_stack((x_sample_train, y_sample_train)))
        _, idx_test = tree.query(np.column_stack((x_sample_test, y_sample_test)))
        lhsdata_train = data_domain_2[idx_train, :]
        lhsdata_test = data_domain_2[idx_test, :]
        # print(lhsdata_train.shape, lhsdata_test.shape)
        data_list_train.append(lhsdata_train)
        data_list_test.append(lhsdata_test)
    data_list_train = np.vstack(data_list_train)
    data_list_test = np.vstack(data_list_test)
    bcinput = np.vstack(bc_input_list)
    bclabel = np.vstack(bc_label_list)
    bcinput = torch.from_numpy(bcinput).float()
    bclabel = torch.from_numpy(bclabel).float()

    t_train = data_list_train[:, 0:1]
    x_train = data_list_train[:, 1:2]
    y_train = data_list_train[:, 2:3]
    u_train = data_list_train[:, 3:4]
    v_train = data_list_train[:, 4:5]
    p_train = data_list_train[:, 5:6]
    train_input = np.concatenate((x_train, y_train, t_train), 1)
    train_label = np.concatenate((u_train, v_train, p_train), 1)
    t_test = data_list_test[:, 0:1]
    x_test = data_list_test[:, 1:2]
    y_test = data_list_test[:, 2:3]
    u_test = data_list_test[:, 3:4]
    v_test = data_list_test[:, 4:5]
    p_test = data_list_test[:, 5:6]
    test_input = np.concatenate((x_test, y_test, t_test), 1)
    test_label = np.concatenate((u_test, v_test, p_test), 1)
    train_input = torch.from_numpy(train_input).float()
    train_label = torch.from_numpy(train_label).float()
    test_input = torch.from_numpy(test_input).float()
    test_label = torch.from_numpy(test_label).float()
    ic_input = torch.from_numpy(ic_input).float()
    ic_label = torch.from_numpy(ic_label).float()
    print(train_input.shape, train_label.shape, bcinput.shape, bclabel.shape)
    return [train_input, test_input, train_label, test_label, ic_input, ic_label, bcinput, bclabel]
# def load_newcylinderdata(datafile,num_train, num_test,num_ic,bc_x,bc_y):
#     data = np.loadtxt(f"{datafile}")
#     data_domain = data
#     # print(data_domain.shape[0]) #4w
#     # choose number of training points: num =7000
#
#
#
#     #train points
#     data0 = data[data[:,0] == 40]
#     idx_ic = np.random.choice(data0.shape[0], num_ic, replace=False)
#     t_ic = data0[idx_ic, 0:1]
#     x_ic = data0[idx_ic, 1:2]
#     y_ic = data0[idx_ic, 2:3]
#     u_ic = data0[idx_ic, 3:4]
#     v_ic = data0[idx_ic, 4:5]
#     p_ic = data0[idx_ic, 5:6]
#     ic_input  = np.concatenate((x_ic, y_ic,t_ic), 1)
#     ic_label = np.concatenate((u_ic,v_ic,p_ic), 1)
#     time_list = np.unique(data_domain[:,0])
#     data_list_train = []
#     data_list_test = []
#     bc_input_list = []
#     bc_label_list = []
#     t_span = len(time_list)
#     n_test = int(num_test/t_span)
#     n_train = int(num_train/t_span)
#     print('test_point_number_for_each_time',n_test,'train_point_number_for_each_time',n_train)
#     for i in time_list:
#         data_domain_2 = data_domain[data_domain[:,0] == i]
#
#         idxr = np.round(data_domain_2[:, 1], 1) == 7.9
#         idxl = np.round(data_domain_2[:, 1], 2) == 0.50
#         idxu = np.round(data_domain_2[:, 2], 2) == 1.96
#         idxd = np.round(data_domain_2[:, 2], 1) == -2.0
#         bc_input_r = data_domain_2[idxr][:,0:3][:, [1, 2, 0]]
#         bc_input_l = data_domain_2[idxl][:,0:3][:, [1, 2, 0]]
#         bc_input_u = data_domain_2[idxu][:,0:3][:, [1, 2, 0]]
#         bc_input_d = data_domain_2[idxd][:,0:3][:, [1, 2, 0]]
#         bc_output_r = data_domain_2[idxr][:,3:]
#         bc_output_l = data_domain_2[idxl][:,3:]
#         bc_output_u = data_domain_2[idxu][:,3:]
#         bc_output_d = data_domain_2[idxd][:,3:]
#         bc_input_r = torch.from_numpy(bc_input_r).float()
#         #print('aaaaa',bc_input_r.shape)
#         bc_input_l = torch.from_numpy(bc_input_l).float()
#         bc_input_u = torch.from_numpy(bc_input_u).float()
#         bc_input_d = torch.from_numpy(bc_input_d).float()
#         bc_output_r = torch.from_numpy(bc_output_r).float()
#         bc_output_l = torch.from_numpy(bc_output_l).float()
#         bc_output_u = torch.from_numpy(bc_output_u).float()
#         bc_output_d = torch.from_numpy(bc_output_d).float()
#         #print(bc_input_r.shape, bc_input_l.shape, bc_input_u.shape, bc_input_d.shape)
#         l_x_max = 7.9
#         l_x_min = 0.5
#         l_y_max = 1.96
#         l_y_min = -2
#         lhsbc_y = (l_y_max-l_y_min)*lhs(1, bc_y)+l_y_min
#         lhsbc_x = (l_x_max-l_x_min)*lhs(1, bc_x)+l_x_min
#
# #         lhsbc_r_idx = int(lhsbc_y * bc_input_r.shape[0])
# #         lhsbc_l_idx = int(lhsbc_y * bc_input_l.shape[0])
# #         lhsbc_u_idx = int(lhsbc_x * bc_input_u.shape[0])
# #         lhsbc_d_idx = int(lhsbc_x * bc_input_d.shape[0])
#
#         # lhsbc_r_idx = lhsbc_y * bc_input_r.shape[0]
#         # lhsbc_l_idx = lhsbc_y * bc_input_l.shape[0]
#         # lhsbc_u_idx = lhsbc_x * bc_input_u.shape[0]
#         # lhsbc_d_idx = lhsbc_x * bc_input_d.shape[0]
#         lhsbc_r_idx = torch.argmin(torch.abs( bc_input_r[:,1] - lhsbc_y),dim=1)
#         lhsbc_l_idx = torch.argmin(torch.abs( bc_input_l[:,1] - lhsbc_y),dim=1)
#         lhsbc_u_idx = torch.argmin(torch.abs( bc_input_u[:,0] - lhsbc_x),dim=1)
#         lhsbc_d_idx = torch.argmin(torch.abs( bc_input_d[:,0] - lhsbc_x),dim=1)
#
#         lhs_input_r = bc_input_r[lhsbc_r_idx]
#         lhs_input_l = bc_input_l[lhsbc_l_idx]
#         lhs_input_u = bc_input_u[lhsbc_u_idx]
#         lhs_input_d = bc_input_d[lhsbc_d_idx]
#
#         # print('2222222222222222222222')
#         # print(lhs_input_r.shape,lhs_input_l.shape,lhs_input_u.shape,lhs_input_d.shape)
#
#         lhs_output_r = bc_output_r[lhsbc_r_idx]
#         lhs_output_l = bc_output_l[lhsbc_l_idx]
#         lhs_output_u = bc_output_u[lhsbc_u_idx]
#         lhs_output_d = bc_output_d[lhsbc_d_idx]
#         #print(lhs_output_d.shape)
#         bc_input = torch.concatenate((lhs_input_r, lhs_input_l, lhs_input_u,lhs_input_d), axis=0)
#         bc_label = torch.concatenate((lhs_output_r, lhs_output_l, lhs_output_u,lhs_output_d), axis=0)
#         #print(bc_input.shape)
#         bc_input_list.append(bc_input)
#         bc_label_list.append(bc_label)
#
#
#         x_min, x_max = data_domain_2[:, 1].min(), data_domain_2[:, 1].max()
#         y_min, y_max = data_domain_2[:, 2].min(), data_domain_2[:, 2].max()
#         lhs_train = latin_hypercube_2d_uniform(n_train)
#         lhs_test = latin_hypercube_2d_uniform(n_test)
#         x_sample_train = x_min + lhs_train[:, 0] * (x_max - x_min)
#         y_sample_train = y_min + lhs_train[:, 1] * (y_max - y_min)
#         x_sample_test = x_min + lhs_test[:, 0] * (x_max - x_min)
#         y_sample_test = y_min + lhs_test[:, 1] * (y_max - y_min)
#         tree = cKDTree(data_domain_2[:, 1:3])
#         _, idx_train = tree.query(np.column_stack((x_sample_train, y_sample_train)))
#         _, idx_test = tree.query(np.column_stack((x_sample_test, y_sample_test)))
#         lhsdata_train = data_domain_2[idx_train, :]
#         lhsdata_test = data_domain_2[idx_test, :]
#         #print(lhsdata_train.shape, lhsdata_test.shape)
#         data_list_train.append(lhsdata_train)
#         data_list_test.append(lhsdata_test)
#     data_list_train = np.vstack(data_list_train)
#     data_list_test = np.vstack(data_list_test)
#     bcinput=np.vstack(bc_input_list)
#     bclabel=np.vstack(bc_label_list)
#     bcinput=torch.from_numpy(bcinput).float()
#     bclabel=torch.from_numpy(bclabel).float()
#
#     t_train = data_list_train[:, 0:1]
#     x_train = data_list_train[:, 1:2]
#     y_train = data_list_train[:, 2:3]
#     u_train = data_list_train[:, 3:4]
#     v_train = data_list_train[:, 4:5]
#     p_train = data_list_train[:, 5:6]
#     train_input = np.concatenate((x_train, y_train,t_train), 1)
#     train_label = np.concatenate((u_train, v_train,p_train), 1)
#     t_test = data_list_test[:, 0:1]
#     x_test = data_list_test[:, 1:2]
#     y_test = data_list_test[:, 2:3]
#     u_test = data_list_test[:, 3:4]
#     v_test = data_list_test[:, 4:5]
#     p_test = data_list_test[:, 5:6]
#     test_input = np.concatenate((x_test, y_test,t_test), 1)
#     test_label = np.concatenate((u_test, v_test,p_test), 1)
#     train_input = torch.from_numpy(train_input).float()
#     train_label = torch.from_numpy(train_label).float()
#     test_input = torch.from_numpy(test_input).float()
#     test_label = torch.from_numpy(test_label).float()
#     ic_input = torch.from_numpy(ic_input).float()
#     ic_label = torch.from_numpy(ic_label).float()
#     print(train_input.shape, train_label.shape, bcinput.shape, bclabel.shape)
#     return [train_input, test_input,train_label, test_label,ic_input,ic_label,bcinput,bclabel]
def load_newcavitydata(datafile,num_train, num_test):
    data = np.loadtxt(f"{datafile}", skiprows=1)

    data_domain = data
    # print(data_domain.shape[0]) #4w
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num_train+num_test, replace=False)
    idx_train = idx[:num_train]
    idx_test = idx[num_train:]
    x_train = data_domain[idx_train, 0:1]
    y_train = data_domain[idx_train, 1:2]
    u_train = data_domain[idx_train, 2:3]
    v_train = data_domain[idx_train, 3:4]
    p_train = data_domain[idx_train, 5:6]
    train_input = np.concatenate((x_train, y_train), 1)
    train_label = np.concatenate((u_train, v_train,p_train), 1)
    x_test = data_domain[idx_test, 0:1]
    y_test = data_domain[idx_test, 1:2]
    u_test = data_domain[idx_test, 2:3]
    v_test = data_domain[idx_test, 3:4]
    p_test = data_domain[idx_test, 5:6]
    test_input = np.concatenate((x_test, y_test), 1)
    test_label = np.concatenate((u_test, v_test,p_test), 1)
    train_input = torch.from_numpy(train_input).float()
    train_label = torch.from_numpy(train_label).float()
    test_input = torch.from_numpy(test_input).float()
    test_label = torch.from_numpy(test_label).float()
    return [train_input, test_input,train_label, test_label]
def load_cavitydata(file):

    data = sio.loadmat(file)

    x = data['X_ref']
    y = data['Y_ref']
    u = data['U_ref']
    v = data['V_ref']
    p = data['P_ref']

    x_star = x.reshape(-1, 1)
    y_star = y.reshape(-1, 1)
    u_star = u.reshape(-1, 1)
    v_star = v.reshape(-1, 1)
    p_star = p.reshape(-1, 1)
   # print(x_star.shape, y_star.shape, u_star.shape, v_star.shape, p_star.shape)#(66049,1)
    x_star = torch.from_numpy(x_star).float()
    y_star = torch.from_numpy(y_star).float()
    u_star = torch.from_numpy(u_star).float()
    v_star = torch.from_numpy(v_star).float()
    p_star = torch.from_numpy(p_star).float()
    data_all = torch.cat((x_star, y_star, u_star, v_star, p_star), dim=1)

    is_nan = torch.isnan(data_all[:, 4])

    data_all = data_all[~is_nan]
    x_star = data_all[:,0].unsqueeze(1)
    y_star = data_all[:,1].unsqueeze(1)
    u_star = data_all[:,2].unsqueeze(1)
    v_star = data_all[:,3].unsqueeze(1)
    p_star = data_all[:,4].unsqueeze(1)


    print(x_star.shape, y_star.shape, u_star.shape, v_star.shape, p_star.shape)
    x_y = torch.cat((x_star, y_star), dim=1)
   # print(x_y.shape)#(66049,2)
    train_point_num = 500
    test_point_num = 1200
    input_data_num = 2
    out_data_num = 3
    train_input_point_list = torch.ones((train_point_num, input_data_num))
    train_label_point_list = torch.ones((train_point_num, out_data_num))
    test_input_point_list = torch.ones((test_point_num, input_data_num))
    test_label_point_list = torch.ones((test_point_num, out_data_num))
    unique_numbers = generate_unique_numbers(x_y.shape[0], train_point_num + test_point_num)
    for i in range(0, train_point_num):
        train_input_point_list[i][0] = x_y[unique_numbers[i]][0]
        train_input_point_list[i][1] = x_y[unique_numbers[i]][1]
        train_label_point_list[i][0] = u_star[unique_numbers[i]][0]
        train_label_point_list[i][1] = v_star[unique_numbers[i]][0]
        train_label_point_list[i][2] = p_star[unique_numbers[i]][0]
        # print(unique_numbers[i])

    for i in range(train_point_num, train_point_num + test_point_num):
        test_input_point_list[i - train_point_num][0] = x_y[unique_numbers[i]][0]
        test_input_point_list[i - train_point_num][1] = x_y[unique_numbers[i]][1]
        test_label_point_list[i - train_point_num][0] = u_star[unique_numbers[i]][0]
        test_label_point_list[i - train_point_num][1] = v_star[unique_numbers[i]][0]
        test_label_point_list[i - train_point_num][2] = p_star[unique_numbers[i]][0]
    dataset = {}
    dataset['train_input'] = train_input_point_list
    dataset['test_input'] = test_input_point_list
    dataset['train_label'] = train_label_point_list
    dataset['test_label'] = test_label_point_list
    return dataset['train_input'], dataset['test_input'], dataset['train_label'], dataset['test_label']
def load_bgdata():
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y
    data_x_t, data_u = gen_testdata()
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

    PDE_input = generate_inputPNT(n_var=2, ranges=[[-1, 1], [0, 1]], pnt_num=800, device=device)
    dataset = {}
    dataset['train_input'] = train_input_point_list.to(device)
    dataset['test_input'] = test_input_point_list.to(device)
    dataset['train_label'] = train_label_point_list.to(device)
    dataset['test_label'] = test_label_point_list.to(device)
    dataset['pde'] = PDE_input.to(device)
    return dataset['train_input'], dataset['test_input'], PDE_input, dataset['train_label'], dataset['test_label']

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


def load_kovdata():
    Kovasznay_Re = 40.
    Kovasznay_nu = 1. / Kovasznay_Re
    Kovasznay_l = 1. / (2. * Kovasznay_nu) - np.sqrt(1. / (4. * Kovasznay_nu ** 2) + 4. * (np.pi ** 2))

    # iterations = 10000

    # create dataset
    f_u = lambda x: 1. - torch.exp(Kovasznay_l * x[:, [0]]) * torch.cos(2. * torch.pi * x[:, [1]])
    dataset_u = create_dataset(f_u, n_var=2, ranges=[[-0.5, 1.0], [-0.5, 1.5]], train_num=675, test_num=3000)

    f_v = lambda x: Kovasznay_l * (2. * torch.pi) ** (-1) * torch.exp(Kovasznay_l * x[:, [0]]) * torch.sin(
        2. * torch.pi * x[:, [1]])
    dataset_v = create_dataset(f_v, n_var=2, ranges=[[-0.5, 1.0], [-0.5, 1.5]], train_num=675, test_num=3000)

    f_p = lambda x: (1. / 2.) * (1. - torch.exp(2. * Kovasznay_l * x[:, [0]]))
    dataset_p = create_dataset(f_p, n_var=2, ranges=[[-0.5, 1.0], [-0.5, 1.5]], train_num=675, test_num=3000)
    # print(dataset_p['train_label'].shape)

    PDE_input = generate_inputPNT(n_var=2, ranges=[[-0.5, 1.0], [-0.5, 1.5]], pnt_num=2601)

    dataset = {}

    dataset['train_input'] = dataset_u['train_input']
    dataset['test_input'] = dataset_u['test_input']
    dataset['train_label'] = torch.cat((dataset_u['train_label'], dataset_v['train_label'], dataset_p['train_label']),
                                       dim=1)
    dataset['test_label'] = torch.cat((dataset_u['test_label'], dataset_v['test_label'], dataset_p['test_label']),
                                      dim=1)
    dataset['pde'] = PDE_input
    return dataset['train_input'], dataset['test_input'], PDE_input, dataset['train_label'], dataset['test_label']

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
