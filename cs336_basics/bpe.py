import os
import regex as re
from collections import defaultdict, Counter
from typing import Iterable, Iterator
import pickle

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




class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # 步骤1: 预分词 - 使用与训练时相同的正则表达式
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # 如果有特殊token，先处理特殊token
        if self.special_tokens:
            # 按特殊token分割文本，优先匹配最长的token
            # 按长度降序排序特殊token
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            
            # 找到所有特殊token的位置
            special_token_positions = []
            for match in re.finditer(pattern, text):
                special_token_positions.append((match.start(), match.end(), match.group()))
            
            # 按位置排序
            special_token_positions.sort()
            
            # 重新构建文本和特殊token序列
            result_ids = []
            current_pos = 0
            
            for start, end, special_token in special_token_positions:
                # 处理特殊token之前的文本
                if start > current_pos:
                    normal_text = text[current_pos:start]
                    result_ids.extend(self._encode_normal_text(normal_text))
                
                # 添加特殊token的ID
                special_token_id = self._get_special_token_id(special_token)
                if special_token_id is not None:
                    result_ids.append(special_token_id)
                
                current_pos = end
            
            # 处理最后剩余的文本
            if current_pos < len(text):
                normal_text = text[current_pos:]
                result_ids.extend(self._encode_normal_text(normal_text))
            
            return result_ids
        else:
            return self._encode_normal_text(text)
    
    def _encode_normal_text(self, text: str) -> list[int]:
        """编码普通文本（不包含特殊token）"""
        # 预分词
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, text)
        
        result_ids = []
        for pre_token in pre_tokens:
            # 将预分词转换为字节序列
            pre_token_bytes = pre_token.encode('utf-8')
            tokens = [bytes([b]) for b in pre_token_bytes]
            
            # 应用BPE合并规则
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and 
                        tokens[i] == pair[0] and 
                        tokens[i + 1] == pair[1]):
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            # 将token转换为ID
            for token in tokens:
                token_id = self._get_token_id(token)
                result_ids.append(token_id)
        
        return result_ids
    
    def _get_token_id(self, token: bytes) -> int:
        """获取token对应的ID"""
        for token_id, vocab_token in self.vocab.items():
            if vocab_token == token:
                return token_id
        # 如果token不在词汇表中，按字节处理
        if len(token) == 1:
            return token[0]
        else:
            # 对于多字节token，按字节分解
            result = []
            for byte in token:
                result.append(byte)
            return result[0] if result else 0
    
    def _get_special_token_id(self, special_token: str) -> int | None:
        """获取特殊token的ID"""
        if self.special_tokens and special_token in self.special_tokens:
            # 在词汇表中查找特殊token
            special_token_bytes = special_token.encode('utf-8')
            for token_id, vocab_token in self.vocab.items():
                if vocab_token == special_token_bytes:
                    return token_id
        return None

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        # 将ID转换为token
        tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                tokens.append(self.vocab[token_id])
            else:
                # 如果ID不在词汇表中，按字节处理
                tokens.append(bytes([token_id]))
        
        # 合并所有token的字节
        result_bytes = b''.join(tokens)
        
        # 解码为字符串
        try:
            return result_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 如果解码失败，使用错误处理
            return result_bytes.decode('utf-8', errors='replace')


