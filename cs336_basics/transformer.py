from jaxtyping import Float
from numpy import dtype
from torch import Tensor
from torch import nn
import torch
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        构造线性变换模块。
        参数:
            in_features (int): 输入的最后一维大小
            out_features (int): 输出的最后一维大小
            device (torch.device | None): 存储参数的设备
            dtype (torch.dtype | None): 参数的数据类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入应用线性变换。
        参数:
            x (torch.Tensor): 输入张量
        返回:
            torch.Tensor: 线性变换后的张量
        """
        return einsum(X, D, D1)



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)

        # 计算RMS，保持最后一个维度以便广播！
        y = torch.sqrt(torch.mean(x**2.0, dim=-1, keepdim=True) + self.eps)

        result = x / y 

        return result.to(device=self.device, dtype=self.dtype)

def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    norm = RMSNorm(d_model, eps, 
        device=in_features.device, dtype=in_features.dtype)

    in_features = norm(in_features)


    return in_features * weights.to(in_features.device)


def silu(x: torch.Tensor):

    return torch.sigmoid(x) * x