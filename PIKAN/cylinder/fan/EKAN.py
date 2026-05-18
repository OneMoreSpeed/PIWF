import torch
import torch.nn.functional as F
import math
import os
import numpy as np
import torch.nn as nn
import deepxde as dde

class EKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        
        spline_order=10,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        batch_size = 100,
    ):
        super(EKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.spline_order = spline_order
        self.batch_size = batch_size
        self.cheby_coeffs = nn.Parameter(torch.empty(out_features, in_features, spline_order+1))
##        h = (grid_range[1] - grid_range[0]) / grid_size
##        grid = (
##            (
##                torch.arange(-spline_order, grid_size + spline_order + 1) * h
##                + grid_range[0]
##            )
##            .expand(in_features, -1)
##            .contiguous()
##        )
        
        gridpre = torch.cos(torch.pi * torch.arange(self.batch_size) / (self.batch_size - 1))
        grid = torch.sort(gridpre).values.expand(in_features, -1).contiguous()
        
        #print('ababab',grid)
        #print('shapshape',grid.shape)
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, spline_order+1)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.batch_size, self.in_features, self.out_features)
                    - 1 / 2
                )
                
                * self.scale_noise
                / self.batch_size
            )
            #print('DDDDDDDDD',noise.shape)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T,
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

##    def b_splines(self, x: torch.Tensor):
##        """
##        Compute the B-spline bases for the given input tensor.
##
##        Args:
##            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
##
##        Returns:
##            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
##        """
##        assert x.dim() == 2 and x.size(1) == self.in_features
##
##        grid: torch.Tensor = (
##            self.grid
##        )  # (in_features, grid_size + 2 * spline_order + 1)
##        #print(x.shape,'pro')
##        #print('ababab',grid.shape)
##        x = x.unsqueeze(-1)
##        #print(x.shape,'aft')
##        ##print(x)
##        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
##        print('aaaaaa',bases.shape,x.shape,grid.shape)#[6,2,11],[6,2,1]/[6,5,11],[6,5,1]-[10,2,11],[10,2,1]/[10,5,11],[10,5,1]
##        #print('aaaa',grid,'bbbbb',bases)
##        for k in range(1, self.spline_order + 1):
##            bases = (
##                (x - grid[:, : -(k + 1)])
##                / (grid[:, k:-1] - grid[:, : -(k + 1)])
##                * bases[:, :, :-1]
##            ) + (
##                (grid[:, k + 1 :] - x)
##                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
##                * bases[:, :, 1:]
##            )
##
##        assert bases.size() == (
##            x.size(0),
##            self.in_features,
##            self.grid_size + self.spline_order,
##        )
##        return bases.contiguous()
    def chebyshev_polynomial(self,x,K=3):
        cheb = torch.ones_like(x)
        if K == 1:
            cheb = x
        elif K > 1:
            cheb_1 = x
            cheb_2 = torch.ones_like(x)
            for k_ in range(1,K):
                cheb =  2 * x * cheb_1-cheb_2
                cheb_2 = cheb_1
                cheb_1 = cheb
        
        return cheb.contiguous()
    
    def chebyshev_bases(self, x: torch.Tensor):
        """
        Compute the chebyshev bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, spline_order+1).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )
        
        # (in_features, grid_size + 2 * spline_order + 1)
        #print(x.shape,'pro')
        
        #x = x.unsqueeze(-1)
        x = torch.tanh(x)
        #print(x.shape,'aft')
        ##print(x)
        #bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        #print('aaaaaa',bases.shape,x.shape)#[6,2,11],[6,2,1]/[6,5,11],[6,5,1]-[10,2,11],[10,2,1]/[10,5,11],[10,5,1]
        #print('aaaa',grid,'bbbbb',bases)
        
        bases = torch.zeros(x.size(0),self.in_features,self.spline_order+1)
        #print('111111111111',bases.shape)
        for k in range(0, self.spline_order + 1):
            bases[:,:,k] = self.chebyshev_polynomial(x,k)
            
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.spline_order+1,
        )
        return bases.contiguous()
        

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        #print('ccccccc',x.size(0), self.in_features, self.out_features,y.size())
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        #print('curve x.shape',x.shape)#[6,2]/[6,5]
        A = self.chebyshev_bases(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        #print('CCCCCCCCC',A.shape,B.shape)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, spline_order+1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.spline_order+1,
        )
        return result.contiguous()

    @property
##    def scaled_spline_weight(self):
##        return self.spline_weight * (
##            self.spline_scaler.unsqueeze(-1)
##            if self.enable_standalone_scale_spline
##            else 1.0
##        )
    def scaled_spline_weight(self):
        return self.cheby_coeffs * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        #print('linear forward.shape',x.shape)#[10,2]/[10,5]
        base_output = F.linear(self.base_activation(x), self.base_weight)
        if torch.cuda.is_available():
            
            device = torch.device('cuda')
        else:
  
            device = torch.device('cpu')
##        spline_output = F.linear(
##            self.chebyshev_bases(x).view(x.size(0), -1),
##            self.scaled_spline_weight.view(self.out_features, -1),
##        )
  

        spline_output = F.linear(
            self.chebyshev_bases(x).view(x.size(0), -1).to(device),
            self.scaled_spline_weight.view(self.out_features, -1).to(device),
        )
        
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

   

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class EKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        
        spline_order=10,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        batch_size=100,
    ):
        super(EKAN, self).__init__()
        
        self.spline_order = spline_order
        self.batch_size = batch_size
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            
            
            self.layers.append(
                EKANLinear(
                    in_features,
                    out_features,
                    
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    batch_size = batch_size
                )
            )

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
        eps = torch.tensor(1e-8, device = self.device, requires_grad=False)

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

    def forward(self, x: torch.Tensor, update_grid=False):
       
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            
            x = layer(x)
           
        return x

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
