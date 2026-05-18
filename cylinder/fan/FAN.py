import torch
import torch.nn.functional as F
import math
import os
import numpy as np
import torch.nn as nn
import deepxde as dde


class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True).
            There is almost no difference between bias and non-bias in our experiments.
    """

    def __init__(self, input_dim, output_dim, p_ratio=0.15, activation='tanh', use_p_bias=True):
        super(FANLayer, self).__init__()

        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        #self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        self.w1_cos = nn.Parameter(torch.ones(p_output_dim, ), requires_grad=True)
        self.w1_sin = nn.Parameter(torch.ones(p_output_dim, ), requires_grad=True)
        self.w2_cos = nn.Parameter(torch.ones(1, ), requires_grad=True)
        self.w2_sin = nn.Parameter(torch.ones(1, ), requires_grad=True)
        self.b_cos = nn.Parameter(torch.zeros(p_output_dim, ), requires_grad=True)
        self.b_sin = nn.Parameter(torch.zeros(p_output_dim, ), requires_grad=True)

        #self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.activation_g = nn.LeakyReLU
    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """

        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))

        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)

        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((self.w2_cos*torch.cos(self.w1_cos*p+self.b_cos), self.w2_sin*torch.sin(self.w1_sin*p+self.b_sin), g), dim=-1)
        #output = torch.cat((self.activation(p), self.activation(p), g), dim=-1)
        return output

class FAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(FAN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

    # Define pdeloss
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

        pred_u, pred_v, pred_p = u[:, [0]], u[:, [1]], u[:,[2]]

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
        eps = torch.tensor(1e-8,  requires_grad=False)

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

        continuity = du_x + dv_y 

        Err = 0.01 * (torch.mean(torch.square(momentum_x)) + torch.mean(torch.square(momentum_y)) + torch.mean(
            torch.square(continuity)))
        # Err_x = 0.1 * torch.mean(torch.square(momentum_x))
        # Err_y = 0.1 * torch.mean(torch.square(momentum_y))
        # Err_continuity = 0.1 * torch.mean(torch.square(continuity))

        dde.grad.clear()
        return Err



    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def Params2File(self,fPath):
        for name, param in self.state_dict().items():
            paramFile = os.path.join(fPath, name)
            fid = open(paramFile, "w")

            params = np.squeeze(param.detach().cpu().numpy())
            print(params)
            if np.ndim(params) <= 1:
                nsize  = np.size(params)
                params = np.reshape(params, (nsize,1))
            np.savetxt(paramFile, params, fmt="%16.7f", delimiter='')
