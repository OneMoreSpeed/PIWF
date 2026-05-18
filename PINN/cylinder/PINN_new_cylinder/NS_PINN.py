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
num_nodes = 128
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
datafile = '../cylinder_nektar_wake.mat'
train_input,test_input,train_label,test_label,ic_input,ic_label,bc_input,bc_label= data.load_newcylinderdata(datafile=datafile,num_train=0,num_test=8000,num_ic=500)
pde_input = generate_inputPNT(n_var = 3, ranges = [[1, 8], [-2, 2],[0, 7]], pnt_num =  140000, device = device)

test_input,pde_input,train_label,test_label,ic_input,ic_label,bc_input,bc_label=test_input.to(device).float() \
    ,pde_input.to(device).float() ,train_label.detach().to(device).float() ,test_label.to(device).float(),\
    ic_input.to(device).float() ,ic_label.to(device).float(),bc_input.to(device).float() ,bc_label.to(device).float()
# variables = torch.FloatTensor(np.concatenate((t_data, x_data), 1)).to(device)
# variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), 1)).to(device)

layer_list = [3] + num_hidden * [num_nodes] + [3]
model = model.pinn(layer_list).to(device)
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
# make save dir

for ep in tqdm(range(num_epochs)):
        weight_bc = 1
        weight_ic = 1
        optimizer.zero_grad()
        
        # Full batch
        #u_hat = model(train_input)
        u_hat_bc = model(bc_input)

        #u_hat_ic = model(ic_input)
        # if eq == 'bg':
        #     loss_f = torch.mean(utils.burgers_equation(u_hat_f, variables_f) ** 2)
        # elif eq == 'ac':
       #loss_f = torch.mean(utils.ac_equation(u_hat_f, variables_f) ** 2)
        loss_pde = model.pdeErr_2D_NS_equation((pde_input))
        loss_bc = torch.mean((u_hat_bc - bc_label) ** 2)
        #loss_u = torch.mean((u_hat - train_label) ** 2)

        #loss_ic = torch.mean((u_hat_ic - ic_label) ** 2)
        loss = loss_pde + loss_bc
        loss.backward() 
        optimizer.step()
        
        l = loss.item()
        loss_graph.append(l)
        with torch.no_grad():
            model.eval()
            output = model(test_input)
            loss_val =  torch.mean((output - test_label) ** 2)

        results['train_loss'].append(loss_bc.cpu().detach().numpy())
        results['pde_loss'].append(loss_pde.cpu().detach().numpy())
        results['test_loss'].append(loss_val.cpu().detach().numpy())
       # results['bc_loss'].append(loss_bc.cpu().detach().numpy())
        #results['ic_loss'].append(loss_ic.cpu().detach().numpy())

        #save weight
        log_Name = "N{}_lr{:.1e}".format(layer_list, lr)
        if ep % 10000 == 0:
            wfile = "weight{}.checkpoint".format(ep)  # 保存模型名称为 weight + epoch数

            cwd = os.getcwd()
            logPath = os.path.join(cwd, log_Name)

            weightFile = os.path.join(logPath, wfile
                                      )
            if not os.path.exists(logPath):
                os.makedirs(logPath)

            torch.save({'epoch': ep, 'tloss': loss_bc, 'ploss': loss_pde, 'vloss': loss_val,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, weightFile)


            
        if ep % 1000 == 0:
            print(f"Train loss: {loss_bc}",f"PDE loss:{loss_pde}",f"VAL loss:{loss_val}")
np.save(f'{layer_list}layer_LBFGS_train_loss.npy', results['train_loss'])
np.save(f'{layer_list}layer_LBFGS_pde_loss.npy', results['pde_loss'])
np.save(f'{layer_list}layer_LBFGS_test_loss.npy',results['test_loss'])
#np.save(f'{layer_list}layer_LBFGS_bc_loss.npy',results['bc_loss'])
#np.save(f'{layer_list}layer_LBFGS_ic_loss.npy',results['ic_loss'])
