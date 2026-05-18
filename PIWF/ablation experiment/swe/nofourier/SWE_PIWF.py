from wavenet.WAVEFAN import WAVEFANBLOCK
from wavenet.WAVEFAN import Net
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
import model

class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=5):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, loss, i):
        loss_aw = 1 / (self.params[i] ** 2) * loss + 1*torch.log(1 + self.params[i] ** 2)
        return loss_aw

awl = AutomaticWeightedLoss(4)
num_epochs = 50001

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
ic_input, ic_label_h, bc_input, bc_label_h, bc_label_u,test_input,test_label_h,test_label_u,pde_input,dataloaders = data.load_SWEdata()

bc_label_h = torch.from_numpy(bc_label_h).float()
test_input = torch.from_numpy(test_input).float().to(device)
test_label_h = torch.from_numpy(test_label_h).float().to(device)
test_label_u = torch.from_numpy(test_label_u).float().to(device)

bc_label_h = bc_label_h.to(device)
#print('aaaaaaaaaa',test_input.shape)
#test_input = test_input.unsqueeze(1)
# test_label_h = test_label_h.unsqueeze(1)
# test_label_u = test_label_u.unsqueeze(1)
input_dim=4
output_dim=2
hidden_dim=50
num_layers=6

#model = Net().to(device)
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
EPOCHS = num_epochs
L1Loss =torch.nn.MSELoss()
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.05, epochs=EPOCHS, steps_per_epoch=len(dataloaders['inner']), pct_start=0.1, div_factor=5, final_div_factor=1000.0)
maxh=bc_label_h.max()
maxu=u=0.29
for ep in tqdm(range(num_epochs)):
    for batch in dataloaders['inner']:
        X = batch
        optimizer.zero_grad()
        # optimize for boundary points
        losses = 0
        
        for batch in dataloaders['initial']:
            x, ic_label_h = batch
            ic_label_h = ic_label_h[:,0]
            u_hat_ic_h,u_hat_ic_u = model(x)[:,0],model(x)[:,1]
            #print('ic_input',x.shape,model(x)[:,0].shape,ic_label_h.shape)
            loss = L1Loss(u_hat_ic_h,ic_label_h)/maxh
            loss_ic = loss
           # loss = awl(loss,0)
            loss.backward()
            losses += loss.item()
        for batch in dataloaders['periodic']: #h
            xb, byh = batch
            u=0.29
            byh=byh[:,0]
            byu = byh*0+u
            bph, bpu = model(xb)[:,0],model(xb)[:,1]
            
            loss = L1Loss(bph,byh)/maxh
            loss_bc_h = loss
            #loss = awl(loss,1)
            loss.backward()
            losses += loss.item() 

        for batch in dataloaders['periodic']: #h
            xb, byh = batch
            u=0.29
            byh=byh[:,0]
            byu = byh*0+u
            bph, bpu = model(xb)[:,0],model(xb)[:,1]
            #print('bc_input',xb.shape,model(xb)[:,0].shape,byu.shape)
            loss = L1Loss(bpu*bph, byu*byh)/(maxu*maxh)
            loss_bc_u = loss
           # loss = awl(loss,2)
            loss.backward()
            losses += loss.item() 
        loss_bc = loss_bc_h+loss_bc_u
        loss_train = loss_bc+loss_ic
        loss = model.pdeErr_SWE(X,bc_label_h.max())
        loss_pde = loss
        #print('pdeinput',X.shape)
        #loss = awl(loss,3)
        
        loss.backward()
        optimizer.step()
        #scheduler.step()
        losses += loss.item()
        
#     optimizer.zero_grad()

#     #u_hat_bc_h ,u_hat_bc_u = model(bc_input)


#     weight_bc =1

#     #loss_pde = model.pdeErr_SWE(pde_input,bc_label_h.max())

#     #loss_ic_h = L1Loss(u_hat_ic_h,ic_label_h)
#     #loss_bc =L1Loss( u_hat_bc_h*u_hat_bc_u,bc_label_h*bc_label_u)
#     loss_bc = loss_bc_h+loss_bc_u
#     loss_train = loss_ic_h+loss_bc
#     loss = loss_pde + loss_train
#     loss.backward() 
#     optimizer.step()

#     l = loss.item()
#     loss_graph.append(l)
    with torch.no_grad():
        model.eval()
        #print('bbbbbbbb',test_input.shape)
        output_h,output_u = model(test_input)[:,0],model(test_input)[:,1]
        output_h[output_h<=0] = 0
        output_u[output_u<=0] = 0
        #print('test_input',test_input.shape,output_h.shape,output_u.shape)
        rmseh = ((output_h-test_label_h)**2).mean()
        rmseu = ((output_u-test_label_u)**2).mean()
        rmseqx =((output_h*output_u-test_label_u*test_label_h)**2).mean()
        loss_val =  torch.mean(( output_h - test_label_h) ** 2)+torch.mean((output_u - test_label_u) ** 2)

    results['train_loss'].append(loss_train.cpu().detach().numpy())
    results['pde_loss'].append(loss_pde.cpu().detach().numpy())
    results['test_loss'].append(loss_val.cpu().detach().numpy())

    #results['bc_loss'].append((loss_bc).cpu().detach().numpy())
    log_Name = "N{}_lr{:.1e}".format(num_layers, lr)
    if ep % 10000 == 0:
        wfile = "weight{}.checkpoint".format(ep)  # 保存模型名称为 weight + epoch数

        cwd = os.getcwd()
        logPath = os.path.join(cwd, log_Name)

        weightFile = os.path.join(logPath, wfile
                                  )
        if not os.path.exists(logPath):
            os.makedirs(logPath)

        torch.save({'epoch': ep, 'rmsh': rmseh,'emsu': rmseqx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, weightFile)



    if ep % 1000 == 0:
        print(f"Train loss: {loss_train}",f"PDE loss:{loss_pde}",f"BC loss:{loss_bc}",f"rmsh:{rmseh}",f"rmsu:{rmseqx}",)
np.save(f'{num_layers}layer_LBFGS_train_loss.npy', results['train_loss'])
np.save(f'{num_layers}layer_LBFGS_pde_loss.npy', results['pde_loss'])
np.save(f'{num_layers}layer_LBFGS_test_loss.npy',results['test_loss'])
#np.save(f'{num_layers}layer_LBFGS_bc_loss.npy',results['bc_loss'])

