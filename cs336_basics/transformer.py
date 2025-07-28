from jaxtyping import Float
from numpy import dtype
from torch import Tensor
from torch import nn
import torch
from einops import einsum

class MyLinear(nn.Module):
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
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入应用线性变换。
        参数:
            x (torch.Tensor): 输入张量
        返回:
            torch.Tensor: 线性变换后的张量
        """
        return einsum(self.weight, x,"d_out d_in, ... d_in -> ... d_out")


class MyEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.empty((vocab_size, d_model), device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        根据token IDs从嵌入矩阵中选择对应的嵌入向量。
        参数:
            token_ids (torch.Tensor): 形状为 (batch_size, sequence_length) 的token ID张量
        返回:
            torch.Tensor: 形状为 (batch_size, sequence_length, d_model) 的嵌入向量张量
        """
        # 使用索引操作从嵌入矩阵中选择对应的嵌入向量
        # self.weight 的形状是 (vocab_size, d_model)
        # token_ids 的形状是 (batch_size, sequence_length)
        # 结果形状将是 (batch_size, sequence_length, d_model)
        return self.weight[token_ids]



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


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    in_features = in_features - in_features.max(dim=dim, keepdim=True)[0]
    exp_in_features = torch.exp(in_features)
    return exp_in_features / exp_in_features.sum(dim=dim, keepdim=True)