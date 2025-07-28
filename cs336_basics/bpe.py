import os
import regex as re
from collections import defaultdict, Counter
import cppyy

# 定义简单的C++函数来优化pair频率统计
cppyy.cppdef("""
#include <map>
#include <string>

std::map<std::string, int> count_pairs_cpp(const std::map<std::string, int>& word_freqs) {
    std::map<std::string, int> pair_freqs;
    
    for (const auto& word_pair : word_freqs) {
        const std::string& word = word_pair.first;
        int freq = word_pair.second;
        
        for (size_t i = 0; i < word.length() - 1; ++i) {
            std::string pair = word.substr(i, 2);
            pair_freqs[pair] += freq;
        }
    }
    
    return pair_freqs;
}
""")

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE tokenizer（使用C++优化）
    
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
    
    # 将每个word转换为字节序列
    word_bytes = {}
    for word in word_freqs:
        word_bytes[word] = word.encode('utf-8')
    
    # 初始化每个word的字节序列
    word_tokens = {}
    for word, freq in word_freqs.items():
        word_tokens[word] = [bytes([b]) for b in word_bytes[word]]
    
    # BPE合并算法
    merges = []
    
    while len(vocab) < vocab_size:
        # 统计所有相邻字节对的频率
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word_tokens[word]
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
        
        # 更新所有words中的pair
        for word in list(word_tokens.keys()):
            tokens = word_tokens[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_tokens[word] = new_tokens
    
    return vocab, merges




