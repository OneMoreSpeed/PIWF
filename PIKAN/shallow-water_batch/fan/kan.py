import torch
import torch.nn.functional as F
import math
import deepxde as dde
import numpy as np
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
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
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
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
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

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


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    def pdeErr_2D_NS_equation(self, x):
        # 设置NS_equation参数
        NS_equation_C1 = dde.Variable(1.0)  #
        NS_equation_C2 = dde.Variable(0.01)  # 粘性系数
        eps = torch.tensor(1e-8, requires_grad=False)

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
    def pdeErr_SWE(self,X,maxh):
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
        maxu=u
        criterion = torch.nn.MSELoss()
        X.requires_grad = True
        ph = self.forward(X)[:,0]
        pu = self.forward(X)[:,1]
        ph[ph<=0] = 0
        pu[pu<=0] = 0
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

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
