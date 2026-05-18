import torch
import torch.nn as nn
import deepxde as dde
import numpy as np
class LinearBlock(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(LinearBlock, self).__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes), dim = 0)

    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x) 
        return x

class PINN(nn.Module):

    def __init__(self, layer_list):
        super(PINN, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_layer = nn.utils.weight_norm(nn.Linear(layer_list[0], layer_list[1]), dim = 0)
        self.hidden_layers = self._make_layer(layer_list[1:-1])
        self.output_layer = nn.Linear(layer_list[-2], layer_list[-1])
        
    def _make_layer(self, layer_list):
        layers = []
        for i in range(len(layer_list) - 1):
            block = LinearBlock(layer_list[i], layer_list[i + 1])
            layers.append(block)
        return nn.Sequential(*layers)



    def pdeErr_1D_Allen_Cahn_equation(self, x):
        d = 0.001
        x.requires_grad_(True)  # x, t
        # print("x", x.shape)
        u = self.forward(x)  # u
        # print("u", u.shape)
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        Err = 0.01 * torch.mean(torch.square(du_t - d * du_xx - 5. * (u - (u ** 3))))
        # print(Err)
        # print("Err", Err.shape)
        dde.grad.clear()

        return Err

    def pdeErr_SWE(self, X, maxh):
        n = 0.03
        u = 0.29
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # define the mesh
        # deta = int(sys.argv[1])
        deta = 10
        nx = 1200 / deta + 1
        ny = 100 / deta + 1
        x = np.linspace(0, 1200, int(nx))
        nt = 21
        t = np.linspace(0, 3600, 21)  # deta t is 600s
        detat = 3600 / (nt - 1)
        maxu = u
        criterion = torch.nn.MSELoss()
        X.requires_grad = True
        ph = self.forward(X)[:, 0]
        pu = self.forward(X)[:, 1]
        ph[ph <= 0] = 0
        pu[pu <= 0] = 0
        pqx = pu * ph
        uqx = pu * pu * ph

        hgrads, = torch.autograd.grad(ph, X,
                                      grad_outputs=ph.data.new(ph.shape).fill_(1),
                                      create_graph=True, only_inputs=True)
        dhdx, dhdt = hgrads[:, 0], hgrads[:, 1]

        ugrads, = torch.autograd.grad(pu, X,
                                      grad_outputs=ph.data.new(ph.shape).fill_(1),
                                      create_graph=True, only_inputs=True)
        dudx, dudt = ugrads[:, 0], ugrads[:, 1]

        dqxdx = dhdx * pu + dudx * ph

        # mass1
        pde1 = (dhdt + dqxdx) / maxh * detat
        # pde1[phs<0.001]=0
        loss1 = (criterion(pde1, pde1 * 0))

        ph = torch.relu(ph)
        dqxdt = dhdt * pu + dudt * ph

        duqxdx = (pu ** 2) * dhdx + 2 * ph * pu * dudx
        # momentum x
        sourcefx = (n ** 2) * pu * ((pu ** 2) ** 0.5)
        # sourcef[torch.abs(phs<0.01)]=0

        pde2 = (dqxdt + duqxdx + 9.81 * ph * dhdx) * ((ph) ** (4 / 3)) + (9.81 * ph * sourcefx)
        pde2 = pde2 * detat / maxu / maxh / maxh ** (4 / 3)
        # pde2[phs<0.001]=0
        # pde = (pde1 + pde2)*1000
        loss2 = (criterion(pde2, pde2 * 0))

        loss = loss1 + loss2
        return loss

    def pdeErr_1D_Burgers_equation(self, x):
        x.requires_grad_(True)  # x, t

        u = self.forward(x)  # u

        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        Err = 0.01 * torch.mean(torch.square(du_t + u * du_x - 0.01 / np.pi * du_xx))
        dde.grad.clear()

        return Err

    def pdeErr_2D_Helmholtz_equation(self, x):
        # 设置Helmholtz参数
        Helmholtz_n = 2.
        Helmholtz_k_0 = 2. * torch.pi * Helmholtz_n
        x.requires_grad_(True)  # x, y
        u = self.forward(x)  # u
        f = torch.sin(Helmholtz_k_0 * x[:, [0]]) * torch.sin(Helmholtz_k_0 * x[:, [1]])
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, i=1, j=1)
        eqn = (du_xx + du_yy) / (Helmholtz_k_0 ** 2) + u + f
        Err = 0.01 * torch.mean(torch.square(eqn))
        # Err = 0.1 * torch.mean(torch.square(du_xx + du_yy + u * Helmholtz_k_0 ** 2 + Helmholtz_k_0 ** 2 * torch.sin(
        #     Helmholtz_k_0 * x[:, 0]) * torch.sin(Helmholtz_k_0 * x[:, 1])))
        # Err = 0.1 * torch.mean(torch.square(du_xx + du_yy + u * Helmholtz_k_0 ** 2 + Helmholtz_k_0 ** 2 * torch.sin(Helmholtz_k_0 * x[:, 0]) * torch.sin(Helmholtz_k_0 * x[:, 1])))
        # print(Err)

        dde.grad.clear()

        return Err

    def pdeErr_2D_Kovasznay_flow(self, x):
        # 设置Kovasznay-flow参数
        Kovasznay_Re = 20.

        x.requires_grad_(True)  # x, y
        u = self.forward(x)  # u, v, p

        pred_u, pred_v, pred_p = u[:, [0]], u[:, [1]], u[:, [2]]

        # 对u的导数
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

        # 对v的导数
        dv_x = dde.grad.jacobian(u, x, i=1, j=0)
        dv_y = dde.grad.jacobian(u, x, i=1, j=1)
        dv_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

        # 对p的导数
        dp_x = dde.grad.jacobian(u, x, i=2, j=0)
        dp_y = dde.grad.jacobian(u, x, i=2, j=1)

        momentum_x = pred_u * du_x + pred_v * du_y + dp_x - 1. / Kovasznay_Re * (du_xx + du_yy)

        momentum_y = pred_u * dv_x + pred_v * dv_y + dp_y - 1. / Kovasznay_Re * (dv_xx + dv_yy)

        continuity = du_x + dv_y

        Err = 0.01 * (torch.mean(torch.square(momentum_x)) + torch.mean(torch.square(momentum_y)) + torch.mean(
            torch.square(continuity)))
        # Err_x = 0.1 * torch.mean(torch.square(momentum_x))
        # Err_y = 0.1 * torch.mean(torch.square(momentum_y))
        # Err_continuity = 0.1 * torch.mean(torch.square(continuity))

        dde.grad.clear()
        return Err

        # return Err, Err_x, Err_y, Err_continuity

    def pdeErr_2D_NS_equation(self, x):
        # 设置NS_equation参数
        NS_equation_C1 = dde.Variable(1.0)  # ？？？意义未知
        NS_equation_C2 = dde.Variable(0.01)  # 粘性系数
        eps = torch.tensor(1e-8, device=self.device, requires_grad=False)

        x.requires_grad_(True)  # x, y, t
        u = self.forward(x)  # u, v, p

        pred_u, pred_v, pred_p = u[:, [0]], u[:, [1]], u[:, [2]]

        # 对u的导数
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_t = dde.grad.jacobian(u, x, i=0, j=2)
        du_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

        # 对v的导数
        dv_x = dde.grad.jacobian(u, x, i=1, j=0)
        dv_y = dde.grad.jacobian(u, x, i=1, j=1)
        dv_t = dde.grad.jacobian(u, x, i=1, j=2)
        dv_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

        # 对p的导数
        dp_x = dde.grad.jacobian(u, x, i=2, j=0)
        dp_y = dde.grad.jacobian(u, x, i=2, j=1)

        # x_momentum = du_t + NS_equation_C1 * (u * du_x + v * du_y) + dp_x - NS_equation_C2 * (du_xx + du_yy)
        # y_momentum = dv_t + NS_equation_C1 * (u * dv_x + v * dv_y) + dp_y - NS_equation_C2 * (dv_xx + dv_yy)

        momentum_x = du_t + NS_equation_C1 * (pred_u * du_x + pred_v * du_y) + dp_x - NS_equation_C2 * (du_xx + du_yy)

        momentum_y = dv_t + NS_equation_C1 * (pred_u * dv_x + pred_v * dv_y) + dp_y - NS_equation_C2 * (dv_xx + dv_yy)

        continuity = du_x + dv_y + eps * pred_p

        Err = 0.01 * (torch.mean(torch.square(momentum_x)) + torch.mean(torch.square(momentum_y)) + torch.mean(
            torch.square(continuity)))
        # Err_x = 0.1 * torch.mean(torch.square(momentum_x))
        # Err_y = 0.1 * torch.mean(torch.square(momentum_y))
        # Err_continuity = 0.1 * torch.mean(torch.square(continuity))

        dde.grad.clear()
        return Err
    def forward(self, x):
        x = self.input_layer(x)
        x = torch.tanh(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def pinn(layer_list):
    model = PINN(layer_list)
    model.apply(weights_init)
    return model
