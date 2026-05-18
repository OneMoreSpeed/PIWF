import torch
import torch.nn.functional as F
import math
import os
import numpy as np
import torch.nn as nn
import deepxde as dde
from sklearn.preprocessing import StandardScaler

class MLPHead(nn.Module):
    """one moded head of self-attention"""

    def __init__(self,channel_size,hidden_size,out_size):
        super().__init__()
        self.nnet = nn.Sequential(
            nn.Linear(channel_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size, bias=False),
        )
        self.value = nn.Linear(channel_size, out_size, bias=False)
        #self.register_buffer("tril", torch.tril(torch.ones(sql, sql)))

        self.dropout = nn.Dropout(0.2)
    def scalablesoftmax(self,s,z_t):
        n = len(z_t)
        scaling_factor = s * torch.log(torch.tensor(n, dtype=z_t.dtype, device=z_t.device))
        z_t_stable = z_t - z_t.max()
        # Numerators: exp((s * log(n)) * z_i)
        numerators = torch.exp(scaling_factor * z_t_stable)

        # Denominator: sum of numerators
        denominator = numerators.sum()

        # Mapped values
        mapped_values = numerators / denominator
       
        return mapped_values

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        wei = self.nnet(x)

        wei = self.scalablesoftmax(1,wei)
       # wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)因为每一个时间步都会产生一个长度为T的向量,所以outsize也设计为t
        # wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = F.sigmoid(self.dropout(wei))
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
       # print('bbbbbb',wei.shape,v.shape)
       # out = wei @ v  # (B, T, T) @ (B, T, Os) -> (B, T, Os)
        out = wei * v
        #print(out.shape)
        return out

class l_channelattention(nn.Module):
    """A 2-layer parameter embedding module for 2D data."""

    def __init__(self,
                 hidden_dim,
                 num_layer=2

                 ):
        super().__init__()
        self.num_layer = num_layer
        self.sql = 2
       # self.fc1 = WAVEFANLayer(hidden_dim, hidden_dim,p_ratio=0.15,q_ratio=0)
        self.fc2 = WAVEFANLayer(hidden_dim, hidden_dim,p_ratio=0,q_ratio=0.3)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(self.num_layer*hidden_dim, hidden_dim)
        self.MLPHead = MLPHead(channel_size=128,hidden_size=256,out_size=128)

    def forward(self, inputs):
        x = inputs
        #print('aaaaa',inputs.shape)
        b, c = x.size()
       # a1 = self.fc1(x)
        a2 = self.fc2(x)
        a3 = self.fc3(x)
        #a1 = a1
        a2 = a2
        a3 = F.gelu(a3)
        y = torch.stack(( a2, a3), dim=1)

        #print('aaaa',y.shape)
        attn = self.MLPHead(y)
        #print('attn',attn.shape)
        pre_output = (y+attn).view(b, -1)
        output = self.proj(pre_output).view(b, c)
        #print('bbbbb',output.shape)
#         proj_attn=self.proj(attn)
#         #print('projattn',proj_attn.shape)

#         pre_output = x+proj_attn
#         output = pre_output.view(b, c)
        return output
class WAVEFANBLOCK(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(WAVEFANBLOCK, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.l_channelattention = l_channelattention(hidden_dim=128)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(l_channelattention(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)

        return output
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
class WaveletBasis:


    def __init__(self, wavelet_type='haar'):
        self.wavelet_type = wavelet_type

    # def wavelet_basis(self, x, j, k, wavelet_type=None):
    #     """ ψ_j,k(x) = 2^(j/2) * ψ(2^j * x - k)"""
    #     if wavelet_type is None:
    #         wavelet_type = self.wavelet_type
    #
    #     scale = 2 ** (j / 2)
    #     shifted_x = 2 ** j * x - k
    #
    #     if wavelet_type == 'haar':
    #
    #         return scale * np.piecewise(shifted_x,
    #                                     [shifted_x < 0,
    #                                      (shifted_x >= 0) & (shifted_x < 0.5),
    #                                      (shifted_x >= 0.5) & (shifted_x < 1),
    #                                      shifted_x >= 1],
    #                                     [0, 1, -1, 0])
    #     elif wavelet_type == 'mexican_hat':
    #
    #         return scale * (1 - shifted_x ** 2) * np.exp(-shifted_x ** 2 / 2)
    #     elif wavelet_type == 'morlet':
    #
    #         return scale * np.cos(5 * shifted_x) * np.exp(-shifted_x ** 2 / 2)
    #     else:
    #         return scale * np.piecewise(shifted_x,
    #                                     [shifted_x < 0,
    #                                      (shifted_x >= 0) & (shifted_x < 0.5),
    #                                      (shifted_x >= 0.5) & (shifted_x < 1),
    #                                      shifted_x >= 1],
    #                                     [0, 1, -1, 0])
    import torch

    def wavelet_basis(self, x, j, k, wavelet_type=None):
        """ ψ_j,k(x) = 2^(j/2) * ψ(2^j * x - k)"""
        if wavelet_type is None:
            wavelet_type = self.wavelet_type


        scale = 2 ** (j / 2)
        shifted_x = 2 ** j * x - k

        if wavelet_type == 'haar':
            # 使用 torch.where 实现分段函数
            zeros = torch.zeros_like(shifted_x)
            ones = torch.ones_like(shifted_x)

            # 创建条件掩码
            mask_neg = shifted_x < 0.0
            mask_low = (shifted_x >= 0.0) & (shifted_x < 0.5)
            mask_mid = (shifted_x >= 0.5) & (shifted_x < 1.0)
            mask_high = shifted_x >= 1.0

            # 应用分段函数
            result = torch.where(mask_neg, zeros,
                                 torch.where(mask_low, ones,
                                             torch.where(mask_mid, -ones,
                                                         torch.where(mask_high, zeros, zeros))))

            return scale * result

        elif wavelet_type == 'mexican_hat':
            # Mexican hat 小波
            shifted_x_sq = torch.square(shifted_x)
            return scale * (1.0 - shifted_x_sq) * torch.exp(-shifted_x_sq / 2.0)

        elif wavelet_type == 'morlet':
            # Morlet 小波
            shifted_x_sq = torch.square(shifted_x)
            return scale * torch.cos(5.0 * shifted_x) * torch.exp(-shifted_x_sq / 2.0)

        else:
            # 默认使用 Haar 小波
            zeros = torch.zeros_like(shifted_x)
            ones = torch.ones_like(shifted_x)

            mask_neg = shifted_x < 0.0
            mask_low = (shifted_x >= 0.0) & (shifted_x < 0.5)
            mask_mid = (shifted_x >= 0.5) & (shifted_x < 1.0)
            mask_high = shifted_x >= 1.0

            result = torch.where(mask_neg, zeros,
                                 torch.where(mask_low, ones,
                                             torch.where(mask_mid, -ones,
                                                         torch.where(mask_high, zeros, zeros))))

            return scale * result
    def scaling_basis(self, x, j, k, wavelet_type=None):
        """ φ_j,k(x) = 2^(j/2) * φ(2^j * x - k)"""
        # if wavelet_type is None:
        #     wavelet_type = self.wavelet_type
        #
        # scale = 2 ** (j / 2)
        # shifted_x = 2 ** j * x - k
        #
        # if wavelet_type == 'haar':
        #     # Haar
        #     return scale * np.piecewise(shifted_x,
        #                                 [shifted_x < 0,
        #                                  (shifted_x >= 0) & (shifted_x < 1),
        #                                  shifted_x >= 1],
        #                                 [0, 1, 0])
        #
        # elif wavelet_type == 'mexican_hat':
        #
        #     return scale * np.exp(-shifted_x ** 2 / 2)
        #
        # elif wavelet_type == 'morlet':
        #
        #     return scale * np.exp(-shifted_x ** 2 / 2)
        #
        # elif wavelet_type == 'db2':
        #
        #     return scale * self._daubechies2_scaling(shifted_x)
        #
        # elif wavelet_type == 'db4':
        #     # Daubechies 4尺度函数
        #     return scale * self._daubechies4_scaling(shifted_x)
        #
        # elif wavelet_type == 'sym2':
        #     # Symlet 2尺度函数
        #     return scale * self._symlet2_scaling(shifted_x)
        #
        # else:
        #     return scale * np.piecewise(shifted_x,
        #                                 [shifted_x < 0,
        #                                  (shifted_x >= 0) & (shifted_x < 1),
        #                                  shifted_x >= 1],
        #                                 [0, 1, 0])

        if wavelet_type is None:
            wavelet_type = self.wavelet_type

        scale = 2 ** (j / 2)
        shifted_x = 2 ** j * x - k

        if wavelet_type == 'haar':
            # Haar小波 - 使用PyTorch实现
            condition = (shifted_x >= 0) & (shifted_x < 1)
            return scale * torch.where(condition, torch.ones_like(shifted_x), torch.zeros_like(shifted_x))

        elif wavelet_type == 'mexican_hat':
            # Mexican hat小波 - 使用PyTorch实现
            return scale * torch.exp(-shifted_x ** 2 / 2)

        elif wavelet_type == 'morlet':
            # Morlet小波 - 使用PyTorch实现
            return scale * torch.exp(-shifted_x ** 2 / 2)

        elif wavelet_type == 'db2':
            # Daubechies 2小波 - 使用PyTorch实现
            return scale * self._daubechies2_scaling_torch(shifted_x)

        elif wavelet_type == 'db4':
            # Daubechies 4小波 - 使用PyTorch实现
            return scale * self._daubechies4_scaling_torch(shifted_x)

        elif wavelet_type == 'sym2':
            # Symlet 2小波 - 使用PyTorch实现
            return scale * self._symlet2_scaling_torch(shifted_x)

        else:
            # 默认使用Haar小波
            condition = (shifted_x >= 0) & (shifted_x < 1)
            return scale * torch.where(condition, torch.ones_like(shifted_x), torch.zeros_like(shifted_x))
    def _daubechies2_scaling(self, x):
        """Daubechies 2尺度函数（近似）"""

        # x = np.clip(x, 0, 3)  # 限制在支撑区间内
        #
        # # 分段线性近似
        # return np.piecewise(x,
        #                     [x < 1, (x >= 1) & (x < 2), x >= 2],
        #                     [lambda t: (1 + np.sqrt(3)) / 4 * t,
        #                      lambda t: (3 + np.sqrt(3)) / 4 - (np.sqrt(3)) / 2 * t,
        #                      lambda t: (3 - np.sqrt(3)) / 4 - (1 - np.sqrt(3)) / 4 * t])
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))

        # 简化的尺度函数近似
        condition1 = (x >= 0) & (x < 1)
        condition2 = (x >= 1) & (x < 2)

        result = torch.zeros_like(x)
        result = torch.where(condition1, h0 + h1 * x, result)
        result = torch.where(condition2, h2 + h3 * (x - 1), result)

        return result


    def _daubechies4_scaling_torch(self, x):
        """Daubechies 4小波尺度函数的PyTorch实现"""
        # Daubechies 4滤波器系数
        h = [
            (1 + np.sqrt(10) + np.sqrt(5 + 2 * np.sqrt(10))) / (16 * np.sqrt(2)),
            (5 + np.sqrt(10) + 3 * np.sqrt(5 + 2 * np.sqrt(10))) / (16 * np.sqrt(2)),
            (10 - 2 * np.sqrt(10) + 2 * np.sqrt(5 + 2 * np.sqrt(10))) / (16 * np.sqrt(2)),
            (10 - 2 * np.sqrt(10) - 2 * np.sqrt(5 + 2 * np.sqrt(10))) / (16 * np.sqrt(2)),
            (5 + np.sqrt(10) - 3 * np.sqrt(5 + 2 * np.sqrt(10))) / (16 * np.sqrt(2)),
            (1 + np.sqrt(10) - np.sqrt(5 + 2 * np.sqrt(10))) / (16 * np.sqrt(2))
        ]

        # 分段线性近似
        result = torch.zeros_like(x)
        for i in range(6):
            condition = (x >= i) & (x < i + 1)
            result = torch.where(condition, torch.full_like(x, h[i]), result)

        return result

class WAVEFANLayer(nn.Module):
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

    def _generate_basis_functions(self):

        basis_funcs = []

        # 添加尺度函数（j=jmax）
        # for k in range(self.k_range[0], self.k_range[1] + 1):
        #     basis_funcs.append({'j': self.j_max, 'k': k, 'type': 'scaling'})
        basis_funcs.append({'j': self.j_max, 'k': 0, 'type': 'scaling'})
        # 添加小波函数（j=0到j_max）
        for j in reversed(range(0, self.j_max + 1)):
            k_min = self.k_range[0] * (2 ** j)
            k_max = self.k_range[1] * (2 ** j)
            for k in range(int(k_min), int(k_max) + 1):
                basis_funcs.append({'j': j, 'k': k, 'type': 'wavelet'})

        return basis_funcs[:5]

    def _compute_basis(self, x, basis_func):

        j, k, func_type = basis_func['j'], basis_func['k'], basis_func['type']
        #print('j=',j, 'k=',k, 'func_type=',func_type)

        if func_type == 'scaling':
            return self.wavelet_basis.scaling_basis(x, j, k, self.wavelet_type)
        else:
            return self.wavelet_basis.wavelet_basis(x, j, k, self.wavelet_type)
    def _compute_design_matrix(self, X):
        #print('aaaaa',X.shape)(10000,7)
        x= X.flatten()
        design_matrices = []
        n_samples = X.shape[0]
        n_basis = len(self.basis_functions)#基函数的数量
        #print(n_basis,'uiybuiuhnln')
        #design_matrix = torch.zeros((n_samples, n_basis))

        design_matrices = [self._compute_basis(x, basis_func) for basis_func in self.basis_functions]
        # for j, basis_func in enumerate(self.basis_functions):
        #     #print('aaaaaa',self._compute_basis(x, basis_func),x)
        #     #design_matrix[i, j] = self._compute_basis(x, basis_func)
        #     matrix_x = self._compute_basis(x, basis_func)
        #     #print(matrix_x.shape)7
        #     design_matrices.append(matrix_x)
        
        design_matrix = torch.cat(design_matrices,dim=-1)
        design_matrix = design_matrix.reshape(n_samples,-1)
        #print('ccccc',design_matrix.shape)
        return design_matrix
    def __init__(self,input_dim, output_dim, p_ratio=0,q_ratio=0.3,j_max=3,k_range=(-2, 2), wavelet_type='haar', activation='gelu', use_p_bias=True):
        super(WAVEFANLayer, self).__init__()

        self.j_max = j_max
        self.k_range = k_range
        self.wavelet_type = wavelet_type

        self.wavelet_basis = WaveletBasis(wavelet_type)

        # 生成小波基函数参数
        self.basis_functions = self._generate_basis_functions()
        self.n_basis = len(self.basis_functions)
        self.coefficients = None
        self.loss_history = []
        # Ensure the p_ratio is within a valid range
        assert 0 <= p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        assert 0 <= q_ratio+2*p_ratio < 1
        self.p_ratio = p_ratio
        self.q_ratio = q_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        q_output_dim = int(output_dim * self.q_ratio)
        g_output_dim = output_dim - p_output_dim * 2-q_output_dim  # Account for cosine and sine terms
        #print('aaaaadadada',p_output_dim,q_output_dim, g_output_dim)
        # Linear transformation for the p component (for cosine and sine parts)
        if self.p_ratio==0:
            self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        else:
            self.input_linear_p = nn.utils.weight_norm(nn.Linear(input_dim, p_output_dim, bias=use_p_bias))
        
        #self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        # Linear transformation for the g component
        self.input_linear_g = nn.utils.weight_norm(nn.Linear(input_dim, g_output_dim))
        if self.q_ratio==0:
            self.input_linear_q = nn.Linear(input_dim*self.n_basis, q_output_dim)
        else:
            self.input_linear_q = nn.utils.weight_norm(nn.Linear(input_dim*self.n_basis, q_output_dim))
        # self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        # self.input_linear_q = nn.Linear(input_dim*self.n_basis, q_output_dim)
        # self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        if p_ratio>0:
            self.w1_cos = nn.Parameter(torch.ones(p_output_dim, ), requires_grad=True)
            self.w1_sin = nn.Parameter(torch.ones(p_output_dim, ), requires_grad=True)
            self.w2_cos = nn.Parameter(torch.ones(1, ), requires_grad=True)
            self.w2_sin = nn.Parameter(torch.ones(1, ), requires_grad=True)
            self.b_cos = nn.Parameter(torch.zeros(p_output_dim, ), requires_grad=True)
            self.b_sin = nn.Parameter(torch.zeros(p_output_dim, ), requires_grad=True)
        self.w_q = nn.Parameter(torch.ones(q_output_dim, ), requires_grad=True)
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
        #print('ddddddd',src.shape)

        #print('fffffff',g.shape,p.shape,q.shape)
        # Concatenate cos(p), sin(p), and activated g along the last dimension
        g = self.activation(self.input_linear_g(src))

        if self.p_ratio==0:
           
            q_wave = self._compute_design_matrix(src)
            q = self.input_linear_q(q_wave)
            if g.dim() == 3 and q.dim() == 2:
        
                q = q.unsqueeze(1)
            elif g.dim() == 2 and q.dim() == 3:
        
                q = q.squeeze(1)
            output = torch.cat((self.w_q*q,g), dim=-1)
        elif self.q_ratio==0:
           
            p = self.input_linear_p(src)
            output = torch.cat((self.w2_cos*torch.cos(self.w1_cos*p+self.b_cos), self.w2_sin*torch.sin(self.w1_sin*p+self.b_sin),g), dim=-1)
        else:
            p = self.input_linear_p(src)
            q_wave = self._compute_design_matrix(src)
            q = self.input_linear_q(q_wave)
            if g.dim() == 3 and q.dim() == 2:
                q = q.unsqueeze(1)
            elif g.dim() == 2 and q.dim() == 3:
                q = q.squeeze(1)
            output = torch.cat((self.w2_cos*torch.cos(self.w1_cos*p+self.b_cos), self.w2_sin*torch.sin(self.w1_sin*p+self.b_sin), self.w_q*q,g), dim=-1)
        #print('bbbbb',output.shape)
        #output = torch.cat((self.activation(p), self.activation(p), g), dim=-1)
        return output

class WAVEFAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(WAVEFAN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.l_channelattention = l_channelattention(hidden_dim=128)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(WAVEFANLayer(hidden_dim, hidden_dim))
        self.layers.append(l_channelattention(hidden_dim))
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
