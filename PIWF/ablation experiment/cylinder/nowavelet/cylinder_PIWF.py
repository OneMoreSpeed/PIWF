import argparse
import torch
import torch.nn as nn
import data
import utils
from wavenet.WAVEFAN import WAVEFANBLOCK
import numpy as np
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
import os
from utils import generate_inputPNT


num_epochs = 100001

lr = 1e-3
eq = 'ac' # or 'bg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Operation mode: ", device)
# if eq == 'bg':
#     t_data, x_data, u_data, t_data_f, x_data_f = data.bg_generator(num_t, num_x)
# elif eq == 'ac':
#     t_data, x_data, u_data, t_data_f, x_data_f = data.ac_generator(num_t, num_x)
# else:
#     print("There exists no the equation.")
#     exit(0)
datafile = './cylinder_nektar_wake.mat'
train_input,test_input,train_label,test_label,ic_input,ic_label,bc_input,bc_label= data.load_newcylinderdata(datafile=datafile,num_train=0,num_test=8000,num_ic=500)
pde_input = generate_inputPNT(n_var = 3, ranges = [[1, 8], [-2, 2],[0, 7]], pnt_num =  140000, device = device)
train_input,test_input,pde_input,train_label,test_label,ic_input,ic_label,bc_input,bc_label=train_input.to(device).float() ,test_input.to(device).float() \
    ,pde_input.to(device).float() ,train_label.detach().to(device).float() ,test_label.to(device).float(),\
    ic_input.to(device).float() ,ic_label.to(device).float(),bc_input.to(device).float() ,bc_label.to(device).float()
# variables = torch.FloatTensor(np.concatenate((t_data, x_data), 1)).to(device)
# variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), 1)).to(device)

input_dim=3
output_dim=3
hidden_dim=128
num_layers=4
model = WAVEFANBLOCK(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.999,0.999), lr=lr)
loss_graph = []
ls = 1e-3
bep = 0
results = {}
results['train_loss'] = []
results['pde_loss'] = []
results['test_loss'] = []
#results['bc_loss'] = []
results['ic_loss'] = []

generator = torch.Generator(device=device)
def create_data_loader(inputs, labels, batch_size, shuffle=True,generator=generator):
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,generator=generator)

batch_size = 5120  # 根据设备调整批次大小
pde_loader = create_data_loader(pde_input, torch.zeros_like(pde_input[:, :1]), batch_size, shuffle=True)
bc_loader = create_data_loader(bc_input, bc_label, batch_size, shuffle=True)
test_loader = create_data_loader(test_input, test_label, batch_size, shuffle=False)

loss_graph = []
results = {'train_loss': [], 'pde_loss': [], 'test_loss': []}

for ep in tqdm(range(num_epochs)):
    weight_bc = 1
    weight_ic = 1
    optimizer.zero_grad()

    total_loss_pde = 0.0
    total_loss_bc = 0.0
    total_loss = 0.0
    batch_count = 0

    for pde_batch, bc_batch in zip(pde_loader, bc_loader):
        pde_input_batch, _ = pde_batch  # pde_loader 的 labels 是全零
        pde_input_batch = pde_input_batch.to(device)

        bc_input_batch, bc_label_batch = bc_batch
        bc_input_batch, bc_label_batch = bc_input_batch.to(device), bc_label_batch.to(device)

        u_hat_bc = model(bc_input_batch)  # 边界条件预测
        loss_bc = torch.mean((u_hat_bc - bc_label_batch) ** 2)  # 边界条件损失

        loss_pde = model.pdeErr_2D_NS_equation(pde_input_batch)  # PDE损失

        loss = loss_pde + loss_bc

        loss.backward()

        optimizer.step()

        total_loss_pde += loss_pde.item()
        total_loss_bc += loss_bc.item()
        total_loss += loss.item()
        batch_count += 1
        optimizer.zero_grad()

    avg_loss_pde = total_loss_pde / batch_count
    avg_loss_bc = total_loss_bc / batch_count
    avg_loss = total_loss / batch_count
    loss_graph.append(avg_loss)

    with torch.no_grad():
        model.eval()
        total_test_loss = 0.0
        test_batch_count = 0
        for test_input_batch, test_label_batch in test_loader:
            test_input_batch, test_label_batch = test_input_batch.to(device), test_label_batch.to(device)
            output = model(test_input_batch)
            loss_val = torch.mean((output - test_label_batch) ** 2)
            total_test_loss += loss_val.item()
            test_batch_count += 1
        avg_test_loss = total_test_loss / test_batch_count

    results['train_loss'].append(avg_loss_bc)
    results['pde_loss'].append(avg_loss_pde)
    results['test_loss'].append(avg_test_loss)

    log_Name = "N{}_lr{:.1e}".format(num_layers, lr)
    if ep % 10000 == 0:
        wfile = "weight{}.checkpoint".format(ep)
        cwd = os.getcwd()
        logPath = os.path.join(cwd, log_Name)
        weightFile = os.path.join(logPath, wfile)

        if not os.path.exists(logPath):
            os.makedirs(logPath)

        torch.save({
            'epoch': ep,
            'tloss': avg_loss_bc,
            'ploss': avg_loss_pde,
            'vloss': avg_test_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weightFile)

    if ep % 1000 == 0:
        print(f"Train loss: {avg_loss_bc:.6f}", f"PDE loss: {avg_loss_pde:.6f}", f"VAL loss: {avg_test_loss:.6f}")
np.save(f'{num_layers}layer_LBFGS_train_loss.npy', results['train_loss'])
np.save(f'{num_layers}layer_LBFGS_pde_loss.npy', results['pde_loss'])
np.save(f'{num_layers}layer_LBFGS_test_loss.npy', results['test_loss'])
