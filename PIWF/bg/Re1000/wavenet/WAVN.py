import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


class WaveletBasis:


    def __init__(self, wavelet_type='haar'):
        self.wavelet_type = wavelet_type

    def wavelet_basis(self, x, j, k, wavelet_type=None):
        """ ψ_j,k(x) = 2^(j/2) * ψ(2^j * x - k)"""
        if wavelet_type is None:
            wavelet_type = self.wavelet_type

        scale = 2 ** (j / 2)
        shifted_x = 2 ** j * x - k

        if wavelet_type == 'haar':

            return scale * np.piecewise(shifted_x,
                                        [shifted_x < 0,
                                         (shifted_x >= 0) & (shifted_x < 0.5),
                                         (shifted_x >= 0.5) & (shifted_x < 1),
                                         shifted_x >= 1],
                                        [0, 1, -1, 0])
        elif wavelet_type == 'mexican_hat':

            return scale * (1 - shifted_x ** 2) * np.exp(-shifted_x ** 2 / 2)
        elif wavelet_type == 'morlet':

            return scale * np.cos(5 * shifted_x) * np.exp(-shifted_x ** 2 / 2)
        else:
            return scale * np.piecewise(shifted_x,
                                        [shifted_x < 0,
                                         (shifted_x >= 0) & (shifted_x < 0.5),
                                         (shifted_x >= 0.5) & (shifted_x < 1),
                                         shifted_x >= 1],
                                        [0, 1, -1, 0])

    def scaling_basis(self, x, j, k, wavelet_type=None):
        """ φ_j,k(x) = 2^(j/2) * φ(2^j * x - k)"""
        if wavelet_type is None:
            wavelet_type = self.wavelet_type

        scale = 2 ** (j / 2)
        shifted_x = 2 ** j * x - k

        if wavelet_type == 'haar':
            # Haar
            return scale * np.piecewise(shifted_x,
                                        [shifted_x < 0,
                                         (shifted_x >= 0) & (shifted_x < 1),
                                         shifted_x >= 1],
                                        [0, 1, 0])

        elif wavelet_type == 'mexican_hat':

            return scale * np.exp(-shifted_x ** 2 / 2)

        elif wavelet_type == 'morlet':

            return scale * np.exp(-shifted_x ** 2 / 2)

        elif wavelet_type == 'db2':

            return scale * self._daubechies2_scaling(shifted_x)

        elif wavelet_type == 'db4':
            # Daubechies 4尺度函数
            return scale * self._daubechies4_scaling(shifted_x)

        elif wavelet_type == 'sym2':
            # Symlet 2尺度函数
            return scale * self._symlet2_scaling(shifted_x)

        else:
            return scale * np.piecewise(shifted_x,
                                        [shifted_x < 0,
                                         (shifted_x >= 0) & (shifted_x < 1),
                                         shifted_x >= 1],
                                        [0, 1, 0])

    def _daubechies2_scaling(self, x):
        """Daubechies 2尺度函数（近似）"""

        x = np.clip(x, 0, 3)  # 限制在支撑区间内

        # 分段线性近似
        return np.piecewise(x,
                            [x < 1, (x >= 1) & (x < 2), x >= 2],
                            [lambda t: (1 + np.sqrt(3)) / 4 * t,
                             lambda t: (3 + np.sqrt(3)) / 4 - (np.sqrt(3)) / 2 * t,
                             lambda t: (3 - np.sqrt(3)) / 4 - (1 - np.sqrt(3)) / 4 * t])
class WaveletSeriesNetwork:


    def __init__(self, j_max=3, k_range=(-2, 2), wavelet_type='haar',
                 learning_rate=0.01, epochs=1000):
        self.j_max = j_max
        self.k_range = k_range
        self.wavelet_type = wavelet_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.wavelet_basis = WaveletBasis(wavelet_type)

        # 生成小波基函数参数
        self.basis_functions = self._generate_basis_functions()
        self.coefficients = None
        self.loss_history = []

    def _generate_basis_functions(self):

        basis_funcs = []

        # 添加尺度函数（j=0）
        for k in range(self.k_range[0], self.k_range[1] + 1):
            basis_funcs.append({'j': 0, 'k': k, 'type': 'scaling'})

        # 添加小波函数（j=0到j_max）
        for j in range(0, self.j_max + 1):
            k_min = self.k_range[0] * np.power(2, j)
            k_max = self.k_range[1] * np.power(2, j)
            for k in range(int(k_min), int(k_max) + 1):
                basis_funcs.append({'j': j, 'k': k, 'type': 'wavelet'})

        return basis_funcs

    def _compute_basis(self, x, basis_func):

        j, k, func_type = basis_func['j'], basis_func['k'], basis_func['type']

        if func_type == 'scaling':
            return self.wavelet_basis.scaling_basis(x, j, k, self.wavelet_type)
        else:
            return self.wavelet_basis.wavelet_basis(x, j, k, self.wavelet_type)

    def _compute_design_matrix(self, X):
        """计算设计矩阵（基函数在数据点上的值）"""
        n_samples = len(X)
        n_basis = len(self.basis_functions)#基函数的数量
        design_matrix = np.zeros((n_samples, n_basis))

        for i, x in enumerate(X):
            for j, basis_func in enumerate(self.basis_functions):
                design_matrix[i, j] = self._compute_basis(x, basis_func)

        return design_matrix

    def fit(self, X, y):
        """训练网络（学习小波系数）"""
        n_basis = len(self.basis_functions)

        # 初始化小波系数
        self.coefficients = np.random.randn(n_basis, 1) * 0.1

        # 计算设计矩阵
        design_matrix = self._compute_design_matrix(X)

        # 使用梯度下降学习系数
        for epoch in range(self.epochs):
            # 前向传播
            predictions = np.dot(design_matrix, self.coefficients)

            # 计算损失
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)

            # 计算梯度
            gradient = 2 * np.dot(design_matrix.T, (predictions - y)) / len(X)

            # 更新系数
            self.coefficients -= self.learning_rate * gradient

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Basis functions: {n_basis}")

        return self

    def predict(self, X):
        """预测"""
        design_matrix = self._compute_design_matrix(X)
        return np.dot(design_matrix, self.coefficients)

    def get_wavelet_coefficients(self):
        """获取小波系数"""
        return self.coefficients

    def get_basis_info(self):
        """获取基函数信息"""
        return self.basis_functions


# 可视化函数
def plot_wavelet_basis(j_values, k_values, wavelet_type='haar'):
    """绘制小波基函数"""
    wavelet_basis = WaveletBasis(wavelet_type)
    x = np.linspace(-2, 2, 1000)

    plt.figure(figsize=(15, 10))

    # 绘制尺度函数
    plt.subplot(2, 1, 1)
    for k in k_values:
        phi = wavelet_basis.scaling_basis(x, 0, k, wavelet_type)
        plt.plot(x, phi, label=f'φ_0,{k}(x)')
    plt.title(f'{wavelet_type.capitalize()} Scaling Functions (j=0)')
    plt.legend()
    plt.grid(True)

    # 绘制小波函数
    plt.subplot(2, 1, 2)
    for j in j_values:
        for k in k_values:
            psi = wavelet_basis.wavelet_basis(x, j, k, wavelet_type)
            plt.plot(x, psi, label=f'ψ_{j},{k}(x)')
    plt.title(f'{wavelet_type.capitalize()} Wavelet Functions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def compare_wavelet_types(X, y, wavelet_types=['haar', 'mexican_hat', 'morlet']):
    """比较不同小波类型的性能"""
    results = {}

    for wavelet_type in wavelet_types:
        print(f"\nTraining with {wavelet_type} wavelet...")

        # 创建小波级数网络
        wsn = WaveletSeriesNetwork(j_max=2, k_range=(-2, 2),
                                   wavelet_type=wavelet_type,
                                   learning_rate=0.01, epochs=500)

        # 训练
        wsn.fit(X, y)

        # 预测
        y_pred = wsn.predict(X)
        mse = np.mean((y_pred - y) ** 2)

        results[wavelet_type] = {
            'model': wsn,
            'mse': mse,
            'coefficients': wsn.coefficients,
            'basis_count': len(wsn.basis_functions)
        }

        print(f"{wavelet_type}: MSE = {mse:.6f}, Basis functions = {len(wsn.basis_functions)}")

    return results


# 主程序
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = np.linspace(-2, 2, 1000).reshape(-1, 1)
    y = np.sin(2 * np.pi * X) + 0.3 * np.cos(5 * np.pi * X) + 0.1 * np.random.randn(1000, 1)

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 可视化小波基函数
    print("Visualizing wavelet basis functions...")
    plot_wavelet_basis(j_values=[0, 1], k_values=[-1, 0, 1], wavelet_type='haar')

    # 比较不同小波类型
    print("\nComparing different wavelet types...")
    results = compare_wavelet_types(X_scaled, y_scaled)

    # 绘制拟合结果
    plt.figure(figsize=(15, 10))

    # 原始数据
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.5, s=1)
    plt.title('Original Data')
    plt.xlabel('X')
    plt.ylabel('y')

    # 不同小波的拟合结果
    colors = ['red', 'blue', 'green']
    for i, (wavelet_type, result) in enumerate(results.items()):
        y_pred = scaler_y.inverse_transform(result['model'].predict(X_scaled))
        plt.subplot(2, 2, 2)
        plt.plot(X, y_pred, color=colors[i], linewidth=2, label=f'{wavelet_type}')

        # 损失曲线
        plt.subplot(2, 2, 3)
        plt.plot(result['model'].loss_history, color=colors[i], label=f'{wavelet_type}')

    plt.subplot(2, 2, 2)
    plt.title('Wavelet Series Approximations')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.yscale('log')

    # 小波系数分布
    plt.subplot(2, 2, 4)
    for i, (wavelet_type, result) in enumerate(results.items()):
        plt.hist(result['coefficients'].flatten(), alpha=0.7,
                 label=f'{wavelet_type}', bins=30)
    plt.title('Wavelet Coefficients Distribution')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印结果
    print("\nFinal Results:")
    for wavelet_type, result in results.items():
        print(f"{wavelet_type:15s}: MSE = {result['mse']:.6f}, "
              f"Basis functions = {result['basis_count']}")
