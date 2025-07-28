import os
import regex as re
from collections import defaultdict, Counter

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE tokenizer
    
    Args:
        input_path: 输入文本文件路径
        vocab_size: 最终词汇表大小（包括字节、合并token和特殊token）
        special_tokens: 特殊token列表
    
    Returns:
        vocab: 词汇表 {token_id: token_bytes}
        merges: 合并列表 [(token1, token2), ...]
    """
    
    # 读取文件
    with open(input_path, "rb") as f:
        content = f.read()
    
    # 移除特殊token，按特殊token分割
    text = content.decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(pattern, text)
    else:
        parts = [text]
    
    # 预分词
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freqs = Counter()
    for part in parts:
        if part:
            tokens = re.findall(PAT, part)
            word_freqs.update(tokens)
    
    # 初始化词汇表：256个字节 + 特殊token
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # 添加特殊token到词汇表
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # 将每个word转换为字节序列并初始化token列表
    word_tokens = {}
    for word, freq in word_freqs.items():
        word_bytes = word.encode('utf-8')
        word_tokens[word] = [bytes([b]) for b in word_bytes]
    
    # BPE合并算法
    merges = []
    
    while len(vocab) < vocab_size:
        # 统计所有相邻字节对的频率 - 优化版本
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word_tokens[word]
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
        
        # 找到频率最高的pair，如果有平局则选择字典序最大的
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        
        # 记录合并
        merges.append(best_pair)
        
        # 创建新token
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # 更新所有words中的pair - 进一步优化
        words_to_update = []
        for word, tokens in word_tokens.items():
            if len(tokens) >= 2:
                # 快速检查是否包含目标pair
                for i in range(len(tokens) - 1):
                    if tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        words_to_update.append(word)
                        break
        
        # 只更新包含目标pair的words
        for word in words_to_update:
            tokens = word_tokens[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == best_pair[0] and 
                    tokens[i + 1] == best_pair[1]):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_tokens[word] = new_tokens
    
    return vocab, merges




