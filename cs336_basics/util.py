from jaxtyping import Float, Int
from torch import Tensor
import torch
from typing import Iterable


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    
    # 减去最大元素以确保数值稳定性
    max_logits = inputs.max(dim=-1, keepdim=True)[0]
    shifted_logits = inputs - max_logits
    
    # 计算 log_softmax，这避免了直接计算 softmax 再取 log
    # log_softmax(x) = x - log(sum(exp(x)))
    log_softmax = shifted_logits - torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
    
    # 获取目标位置的 log_softmax 值
    target_log_softmax = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # 交叉熵损失 = -log_softmax
    return -target_log_softmax.mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 收集所有非空梯度
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    
    if not grads:
        return
    
    # 计算所有梯度的总L2范数，而不是计算一个参数一个参数地算梯度范数！
    total_norm = torch.cat(grads).norm(2)
    
    eps = 1e-6
    clip_coef = max_l2_norm / (total_norm + eps)
    
    # 对所有梯度进行缩放
    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)