from jaxtyping import Float, Int
from numpy import dtype
from torch import Tensor
from torch import nn
import torch
import math
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


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    w1x = einsum(w1_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
    w3x = einsum(w3_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")

    output = silu(w1x) * w3x

    return einsum(w2_weight, output, "d_model d_ff, ... d_ff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 预计算sin和cos值
        # 创建位置索引 [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # 频率下标 - 正确的RoPE频率计算
        freqs = self.theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = positions[:, None] / freqs[None, :]  # (max_seq_len, d_k//2)
        sin_values = torch.sin(angles)
        cos_values = torch.cos(angles)
        self.register_buffer("sin_values", sin_values, persistent=False)
        self.register_buffer("cos_values", cos_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用RoPE旋转到输入张量。
        
        参数:
            x (torch.Tensor): 形状为 (... seq_len, d_k) 的输入张量
            token_positions (torch.Tensor): 形状为 (... seq_len) 的token位置张量
            
        返回:
            torch.Tensor: 旋转后的张量，形状与输入相同
        """
        # 获取输入的形状信息
        original_shape = x.shape
        d_k = self.d_k
        
        # 确保d_k是偶数
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        
        # 重塑输入为 (... seq_len, d_k//2, 2)
        # 这样每对相邻的元素可以作为一个2D向量进行旋转
        x_reshaped = x.view(*original_shape[:-1], d_k // 2, 2)
        
        # 根据token_positions获取对应的sin和cos值
        # token_positions的形状: (... seq_len)
        # 需要广播到 (... seq_len, d_k//2)
        
        # 从预计算的缓冲区中获取sin和cos值
        sin_vals = self.sin_values[token_positions]  # 形状: (... seq_len, d_k//2)
        cos_vals = self.cos_values[token_positions]  # 形状: (... seq_len, d_k//2)
        
        # 应用2D旋转矩阵:
        # [cos(θ)  -sin(θ)] [x1] = [x1*cos(θ) - x2*sin(θ)]
        # [sin(θ)   cos(θ)] [x2]   [x1*sin(θ) + x2*cos(θ)]
        
        # 分离x1和x2（每对的第一个和第二个元素）
        x1 = x_reshaped[..., 0]  # 形状: (... seq_len, d_k//2)
        x2 = x_reshaped[..., 1]  # 形状: (... seq_len, d_k//2)
        
        # 应用旋转
        rotated_x1 = x1 * cos_vals - x2 * sin_vals
        rotated_x2 = x1 * sin_vals + x2 * cos_vals 
        
        # 重新组合旋转后的值
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)  # 形状: (... seq_len, d_k//2, 2)
        
        # 重塑回原始形状
        result = rotated_x.view(original_shape)
        
        return result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scale = 1 / math.sqrt(d_k)

    S = scale * einsum(Q, K.transpose(-2, -1), "... q d_k, ... d_k k -> ... q k")
    if mask is not None:
        S = S.masked_fill(~mask, float("-inf"))
    return einsum(softmax(S, dim=-1), V, "... q k, ... k d_v -> ... q d_v")


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    
    Q = einsum(q_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    K = einsum(k_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    V = einsum(v_proj_weight, in_features,
         "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

    d_k = K.shape[-1]
    seq_len = K.shape[-2]
    d_v = V.shape[-1]

    h = num_heads
    d_kh = d_k // h
    d_vh = d_v // h

    mask = torch.triu(torch.ones(*K.shape[:-2], seq_len, seq_len), diagonal=1).to(device=Q.device, dtype=torch.bool)
    attns = []
    for i in range(h):
        attns.append(scaled_dot_product_attention(
        Q[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        K[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        V[..., i*d_vh: min((i+1)*d_vh, d_v)], 
        mask=~mask))
    
    features = torch.concat(attns, dim=-1)

    return einsum(o_proj_weight, features,
         "d_model d_v, ... sequence_length d_v -> ... sequence_length d_model")


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:

    Q = einsum(q_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    K = einsum(k_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    V = einsum(v_proj_weight, in_features,
         "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

    d_k = K.shape[-1]
    seq_len = K.shape[-2]
    d_v = V.shape[-1]

    h = num_heads
    d_kh = d_k // h
    d_vh = d_v // h

    # 应用RoPE
    if token_positions is not None:
        max_seq_len = token_positions.max().item() + 1
        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_kh,
            max_seq_len=max_seq_len,
            device=in_features.device
        )

        # 对每个头分别应用RoPE
        for i in range(h):
            # 获取当前头的Q和K
            q_head = Q[..., i*d_kh:(i+1)*d_kh]
            k_head = K[..., i*d_kh:(i+1)*d_kh]
            
            # 应用RoPE
            q_head_rope = rope(q_head, token_positions)
            k_head_rope = rope(k_head, token_positions)
            
            # 更新Q和K
            Q[..., i*d_kh:(i+1)*d_kh] = q_head_rope
            K[..., i*d_kh:(i+1)*d_kh] = k_head_rope

    mask = torch.triu(torch.ones(*K.shape[:-2], seq_len, seq_len), diagonal=1).to(device=Q.device, dtype=torch.bool)
    attns = []
    for i in range(h):
        attns.append(scaled_dot_product_attention(
        Q[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        K[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        V[..., i*d_vh: min((i+1)*d_vh, d_v)], 
        mask=~mask))
    
    features = torch.concat(attns, dim=-1)

    return einsum(o_proj_weight, features,
         "d_model d_v, ... sequence_length d_v -> ... sequence_length d_model")


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    



def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]: