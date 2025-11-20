#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
周杰伦歌词RNN生成模型 - PyTorch实现
基于循环神经网络训练语言模型，并生成歌词
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import time
from collections import Counter


# ============================================================================
# 数据预处理
# ============================================================================

class LyricsDataset(Dataset):
    """周杰伦歌词数据集"""

    def __init__(self, text, seq_length=50):
        self.seq_length = seq_length
        self.text = text

        # 构建字符到索引的映射
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"数据集大小: {len(text)} 个字符")
        print(f"词汇表大小: {self.vocab_size} 个不同字符")

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        # 输入序列
        input_seq = self.text[idx:idx + self.seq_length]
        # 目标序列（向后偏移一个字符）
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]

        # 转换为索引
        input_indices = torch.tensor([self.char_to_idx[ch] for ch in input_seq], dtype=torch.long)
        target_indices = torch.tensor([self.char_to_idx[ch] for ch in target_seq], dtype=torch.long)

        return input_indices, target_indices


def load_lyrics_data(data_path='dataset/jaychou_lyrics.txt'):
    """
    加载周杰伦歌词数据
    如果文件不存在，则使用示例数据
    """
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"从 {data_path} 加载歌词数据")
    else:
        print(f"未找到 {data_path}，使用示例数据")
        # 示例歌词数据（周杰伦风格）
        text = """想要有直升机
想要和你飞到宇宙去
想要和你融化在一起
融化在宇宙里
我每天每天每天在想你
这样的甜蜜
让我开始相信命运
感谢地心引力
让我碰到你
漂亮的让我面红的可爱女人
温柔的让我心疼的可爱女人
透明的让我感动的可爱女人
坏坏的让我疯狂的可爱女人
慢慢的你会爱上我
这样的感觉我很确定
不知不觉你会爱上我
我给你的爱写在西元前
深埋在美索不达米亚平原
几十个世纪后出土发现
泥板上的字迹依然清晰可见
我给你的爱写在西元前
深埋在美索不达米亚平原
用楔形文字刻下了永远
那已风化千年的誓言还依然清晰可见
晴天 阴天 雨天
聊天
等你下课回家的时间
手牵手一步两步三步四步望着天
看星星一颗两颗三颗四颗连成线
背对背默默许下心愿
看远方的星是否听得见
一口一口吃掉悲伤
我唱歌你听着我唱
淋着雨一直走到世界尽头
我要送你九十九朵玫瑰
不知不觉我跟了这节奏
后知后觉又过了一个秋
你说你也很爱我
我想带你骑单车
我想和你看棒球
想这样没担忧
唱着歌一直走
我想就这样牵着你的手不放开
爱可不可以简简单单没有伤害
你靠着我的肩膀
你在我胸口睡着
像这样的生活我爱你
你爱我
在枫叶飘落之前
让我再与你相见
为何此刻风吹着
我的心寂寞如雪
听妈妈的话别让她受伤
想快快长大才能保护她
美丽的白发幸福中发芽
天使的魔法温暖中慈祥
我会听妈妈的话
长大后的我开始懂了她
她的白发呀慢慢的长啊
我记住她的魔法
"""
        # 创建dataset目录
        os.makedirs('dataset', exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"已将示例数据保存到 {data_path}")

    return text


# ============================================================================
# RNN模型定义
# ============================================================================

class RNNLyricsModel(nn.Module):
    """
    循环神经网络歌词生成模型
    支持 LSTM 和 GRU
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3, rnn_type='LSTM'):
        super(RNNLyricsModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN层
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")

        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        前向传播
        x: (batch_size, seq_length)
        """
        # 嵌入
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)

        # RNN
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)

        # 输出层
        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, seq_length, vocab_size)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h0, c0)
        else:  # GRU
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return h0


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=5.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_chars = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs, _ = model(inputs)

        # 计算损失
        # outputs: (batch_size, seq_length, vocab_size)
        # targets: (batch_size, seq_length)
        loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0) * inputs.size(1)
        total_chars += inputs.size(0) * inputs.size(1)

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_chars
            perplexity = np.exp(avg_loss)
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} Perplexity: {perplexity:.2f}")

    avg_loss = total_loss / total_chars
    return avg_loss


# ============================================================================
# 歌词生成函数
# ============================================================================

def generate_lyrics(model, dataset, start_string="我", length=200,
                   temperature=1.0, device='cpu'):
    """
    生成歌词

    Args:
        model: 训练好的模型
        dataset: 数据集对象（用于获取字符映射）
        start_string: 起始字符串
        length: 生成的字符数
        temperature: 温度参数，越高生成越随机
        device: 设备
    """
    model.eval()

    with torch.no_grad():
        # 转换起始字符串为索引
        chars = [dataset.char_to_idx.get(c, 0) for c in start_string]
        input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

        # 初始化隐藏状态
        hidden = model.init_hidden(1, device)

        # 生成歌词
        generated = start_string

        for _ in range(length):
            # 前向传播
            output, hidden = model(input_seq, hidden)

            # 取最后一个时间步的输出
            output = output[:, -1, :] / temperature

            # 应用softmax获得概率分布
            probs = torch.softmax(output, dim=-1)

            # 采样下一个字符
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = dataset.idx_to_char[next_char_idx]

            generated += next_char

            # 更新输入序列
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

        return generated


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 超参数
    seq_length = 50          # 序列长度
    batch_size = 64          # 批大小
    embedding_dim = 128      # 嵌入维度
    hidden_dim = 256         # 隐藏层维度
    num_layers = 2           # RNN层数
    dropout = 0.3            # Dropout率
    rnn_type = 'LSTM'        # RNN类型: 'LSTM' 或 'GRU'
    learning_rate = 0.002    # 学习率
    num_epochs = 100         # 训练轮数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("周杰伦歌词RNN生成模型")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"RNN类型: {rnn_type}")
    print(f"序列长度: {seq_length}")
    print(f"批大小: {batch_size}")
    print(f"嵌入维度: {embedding_dim}")
    print(f"隐藏维度: {hidden_dim}")
    print(f"RNN层数: {num_layers}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {num_epochs}")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    text = load_lyrics_data()

    # 创建数据集
    dataset = LyricsDataset(text, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, drop_last=True)

    # 创建模型
    print("\n创建模型...")
    model = RNNLyricsModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print("\n开始训练...")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练一个epoch
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        perplexity = np.exp(train_loss)

        epoch_time = time.time() - start_time

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] "
              f"Loss: {train_loss:.4f} "
              f"Perplexity: {perplexity:.2f} "
              f"Time: {epoch_time:.2f}s")

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_dir = 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char,
                'vocab_size': dataset.vocab_size,
            }, f'{checkpoint_dir}/rnn_lyrics_best.pth')
            print(f"保存最佳模型 (Loss: {train_loss:.4f})")

        # 每10个epoch生成一次歌词
        if (epoch + 1) % 10 == 0:
            print("\n生成歌词示例:")
            print("-" * 80)
            for start_str in ["我", "想", "你", "爱"]:
                generated = generate_lyrics(model, dataset, start_string=start_str,
                                          length=100, temperature=0.8, device=device)
                print(f"起始字符 '{start_str}':")
                print(generated)
                print()
            print("-" * 80)

    print("\n训练完成！")

    # 最终生成歌词
    print("\n" + "=" * 80)
    print("最终生成歌词:")
    print("=" * 80)

    model.eval()
    for temp in [0.5, 0.8, 1.0, 1.2]:
        print(f"\n温度参数 = {temp}:")
        print("-" * 80)
        generated = generate_lyrics(model, dataset, start_string="我想要",
                                   length=200, temperature=temp, device=device)
        print(generated)
        print("-" * 80)

    print("\n模型已保存到 checkpoints/rnn_lyrics_best.pth")
    print("\n可以使用以下代码加载模型并生成歌词:")
    print("""
# 加载模型
checkpoint = torch.load('checkpoints/rnn_lyrics_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 生成歌词
generated = generate_lyrics(model, dataset, start_string="我", length=200, temperature=0.8)
print(generated)
    """)


if __name__ == '__main__':
    main()
