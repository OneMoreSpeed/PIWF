from wavenet.WAVEFAN import WAVEFANBLOCK
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

num_epochs = 200001

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
train_input,test_input,pde_input,train_label,test_label,bc_input,bc_label,ic_input,ic_label = data.load_bgdata()
ic_input = np.squeeze(ic_input,1)
bc_input = np.squeeze(bc_input,1)
ic_label = np.squeeze(ic_label,1)
bc_label = np.squeeze(bc_label,1)
train_input,test_input,pde_input,train_label,test_label,bc_input,bc_label,ic_input,ic_label=train_input.to(device),test_input.to(device),\
    pde_input.to(device),train_label.detach().to(device),test_label.to(device),bc_input.to(device),bc_label.to(device),ic_input.to(device),ic_label.to(device)
# variables = torch.FloatTensor(np.concatenate((t_data, x_data), 1)).to(device)
# variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), 1)).to(device)

input_dim=2
output_dim=1
hidden_dim=50
num_layers=6
model = WAVEFANBLOCK(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.999,0.999), lr=lr)
loss_graph = []
ls = 1e-3
bep = 0
results = {}
results['train_loss'] = []
results['pde_loss'] = []
results['test_loss'] = []
results['bc_loss'] = []
# make save dir

for ep in tqdm(range(num_epochs)):
        
        optimizer.zero_grad()
        #print('aaaaa',bc_input.shape,ic_input.shape,pde_input.shape,ic_label.shape,bc_label.shape)
        # Full batch
        #u_hat = model(train_input)
        u_hat_f = model(pde_input)
        u_hat_bc = model(bc_input)
        u_hat_ic = model(ic_input)
        weight_bc =1
        # if eq == 'bg':
        #     loss_f = torch.mean(utils.burgers_equation(u_hat_f, variables_f) ** 2)
        # elif eq == 'ac':
       #loss_f = torch.mean(utils.ac_equation(u_hat_f, variables_f) ** 2)
        loss_pde = model.pdeErr_1D_Burgers_equation((pde_input))
        #loss_u = torch.mean((u_hat - train_label) ** 2)
        loss_ic = torch.mean((u_hat_ic - ic_label) ** 2)
        loss_bc = torch.mean((u_hat_bc - bc_label) ** 2)
        loss_train = loss_ic+loss_bc
        loss = loss_pde + loss_train
        loss.backward() 
        optimizer.step()
        
        l = loss.item()
        loss_graph.append(l)
        with torch.no_grad():
            model.eval()
            output = model(test_input)
            loss_val =  torch.mean((output - test_label) ** 2)

        results['train_loss'].append(loss_train.cpu().detach().numpy())
        results['pde_loss'].append(loss_pde.cpu().detach().numpy())
        results['test_loss'].append(loss_val.cpu().detach().numpy())

        results['bc_loss'].append(loss_bc.cpu().detach().numpy())
        log_Name = "N{}_lr{:.1e}".format(num_layers, lr)
        if ep % 10000 == 0:
            wfile = "weight{}.checkpoint".format(ep)  # 保存模型名称为 weight + epoch数

            cwd = os.getcwd()
            logPath = os.path.join(cwd, log_Name)

            weightFile = os.path.join(logPath, wfile
                                      )
            if not os.path.exists(logPath):
                os.makedirs(logPath)

            torch.save({'epoch': ep, 'tloss': loss_train, 'ploss': loss_pde, 'vloss': loss_val,'bc_loss': loss_bc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, weightFile)


            
        if ep % 1000 == 0:
            print(f"Train loss: {loss_train}",f"PDE loss:{loss_pde}",f"VAL loss:{loss_val}",f"BC loss:{loss_bc}")
np.save(f'{num_layers}layer_LBFGS_train_loss.npy', results['train_loss'])
np.save(f'{num_layers}layer_LBFGS_pde_loss.npy', results['pde_loss'])
np.save(f'{num_layers}layer_LBFGS_test_loss.npy',results['test_loss'])
np.save(f'{num_layers}layer_LBFGS_bc_loss.npy',results['bc_loss'])

