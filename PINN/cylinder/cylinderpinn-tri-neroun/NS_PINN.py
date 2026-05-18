import argparse
import torch
import torch.nn as nn
import data
import utils
import model
import numpy as np
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
import os
from utils import generate_inputPNT

num_epochs = 100001
num_hidden = 4
num_nodes = 384
lr = 1e-3
eq = 'ac'  # or 'bg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Operation mode: ", device)

datafile = './cylinder_nektar_wake.mat'
train_input, test_input, train_label, test_label, ic_input, ic_label, bc_input, bc_label = data.load_newcylinderdata(
    datafile=datafile, num_train=0, num_test=8000, num_ic=500
)
pde_input = generate_inputPNT(n_var=3, ranges=[[1, 8], [-2, 2], [0, 7]], pnt_num=140000, device=device)

# 转移到 GPU
test_input = test_input.to(device).float()
pde_input = pde_input.to(device).float()
train_label = train_label.detach().to(device).float()
test_label = test_label.to(device).float()
ic_input = ic_input.to(device).float()
ic_label = ic_label.to(device).float()
bc_input = bc_input.to(device).float()
bc_label = bc_label.to(device).float()

layer_list = [3] + num_hidden * [num_nodes] + [3]
model = model.pinn(layer_list).to(device)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.999, 0.999), lr=lr)

# ---------- 新增：批训练超参数 ----------
pde_batch_size = 2000        # 根据 GPU 显存调整，建议 1000~4000
use_gradient_accumulation = True   # 梯度累积

loss_graph = []
ls = 1e-3
bep = 0
results = {}
results['train_loss'] = []
results['pde_loss'] = []
results['test_loss'] = []
results['ic_loss'] = []

# 创建保存目录
log_Name = "N{}_lr{:.1e}".format(layer_list, lr)
cwd = os.getcwd()
logPath = os.path.join(cwd, log_Name)
if not os.path.exists(logPath):
    os.makedirs(logPath)

for ep in tqdm(range(num_epochs)):
    model.train()
    optimizer.zero_grad()

    # ---------- 1. PDE 损失：分批计算并累积梯度 ----------
    total_pde_loss = 0.0
    total_pnts = pde_input.shape[0]
    for start in range(0, total_pnts, pde_batch_size):
        end = min(start + pde_batch_size, total_pnts)
        batch_input = pde_input[start:end]
        loss_batch = model.pdeErr_2D_NS_equation(batch_input)  # 标量，通常是该 batch 的 MSE
        weight = len(batch_input) / total_pnts
        weighted_loss = loss_batch * weight

        # 记录平均 loss（用于日志）
        total_pde_loss += weighted_loss.detach().item()

        # 反向传播，累积梯度（不清零）
        if use_gradient_accumulation:
            weighted_loss.backward()
        else:
            # 非累积模式：立即更新参数（不推荐，会过于频繁）
            weighted_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 若不使用梯度累积，则 PDE 部分已经更新过参数，下面的 BC 部分需单独处理
    # 但通常使用累积，让 PDE 和 BC 梯度合并后再更新

    # ---------- 2. BC 损失（通常较小，可直接全量计算） ----------
    u_hat_bc = model(bc_input)
    loss_bc = torch.mean((u_hat_bc - bc_label) ** 2)
    if use_gradient_accumulation:
        loss_bc.backward()
    else:
        loss_bc.backward()
        optimizer.step()
        optimizer.zero_grad()

    # ---------- 3. 参数更新（仅当使用梯度累积时） ----------
    if use_gradient_accumulation:
        optimizer.step()
        optimizer.zero_grad()

    # ---------- 4. 记录损失（PDE 损失记录为加权平均值） ----------
    loss_pde_value = total_pde_loss  # 已经是 float
    loss_bc_value = loss_bc.item()

    # 总损失（用于 loss_graph 记录）
    total_loss = loss_pde_value + loss_bc_value
    loss_graph.append(total_loss)

    # ---------- 5. 验证集评估 ----------
    with torch.no_grad():
        model.eval()
        output = model(test_input)
        loss_val = torch.mean((output - test_label) ** 2).item()

    # 保存历史记录
    results['train_loss'].append(loss_bc_value)
    results['pde_loss'].append(loss_pde_value)
    results['test_loss'].append(loss_val)

    # 定期保存模型
    if ep % 10000 == 0:
        weightFile = os.path.join(logPath, f"weight{ep}.checkpoint")
        torch.save({
            'epoch': ep,
            'tloss': loss_bc_value,
            'ploss': loss_pde_value,
            'vloss': loss_val,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weightFile)

    if ep % 1000 == 0:
        print(f"Epoch {ep}: BC loss = {loss_bc_value:.4e}, PDE loss = {loss_pde_value:.4e}, Val loss = {loss_val:.4e}")

# 保存损失曲线
np.save(f'{layer_list}layer_LBFGS_train_loss.npy', results['train_loss'])
np.save(f'{layer_list}layer_LBFGS_pde_loss.npy', results['pde_loss'])
np.save(f'{layer_list}layer_LBFGS_test_loss.npy', results['test_loss'])