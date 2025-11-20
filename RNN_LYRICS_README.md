# 周杰伦歌词RNN生成模型

基于PyTorch实现的循环神经网络（RNN）语言模型，用于训练和生成周杰伦风格的歌词。

## 功能特点

- ✅ 完整的单文件实现
- ✅ 支持 LSTM 和 GRU 两种RNN架构
- ✅ 字符级语言模型
- ✅ 自动数据预处理
- ✅ 温度采样控制生成多样性
- ✅ 训练过程可视化
- ✅ 模型自动保存

## 环境要求

```bash
pip install torch numpy
```

## 快速开始

### 1. 运行训练和生成

```bash
python rnn_lyrics_generation.py
```

程序会自动：
- 加载或创建周杰伦歌词数据集
- 构建字符级词汇表
- 训练RNN模型
- 定期生成歌词示例
- 保存最佳模型到 `checkpoints/rnn_lyrics_best.pth`

### 2. 使用自己的歌词数据

将歌词文本放在 `dataset/jaychou_lyrics.txt`，程序会自动加载。

## 模型架构

```
输入文本 (字符序列)
    ↓
嵌入层 (Embedding)
    ↓
RNN层 (LSTM/GRU × 2层)
    ↓
全连接层
    ↓
输出 (字符概率分布)
```

## 超参数配置

可以在 `main()` 函数中修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seq_length` | 50 | 序列长度 |
| `batch_size` | 64 | 批大小 |
| `embedding_dim` | 128 | 嵌入维度 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `num_layers` | 2 | RNN层数 |
| `rnn_type` | 'LSTM' | RNN类型 (LSTM/GRU) |
| `learning_rate` | 0.002 | 学习率 |
| `num_epochs` | 100 | 训练轮数 |

## 生成歌词

### 训练中自动生成

程序每10个epoch会自动生成歌词示例。

### 使用已保存的模型

```python
import torch
from rnn_lyrics_generation import RNNLyricsModel, LyricsDataset, generate_lyrics

# 加载模型
checkpoint = torch.load('checkpoints/rnn_lyrics_best.pth')

# 创建数据集对象（用于字符映射）
text = "..."  # 你的训练数据
dataset = LyricsDataset(text)
dataset.char_to_idx = checkpoint['char_to_idx']
dataset.idx_to_char = checkpoint['idx_to_char']
dataset.vocab_size = checkpoint['vocab_size']

# 创建模型
model = RNNLyricsModel(vocab_size=dataset.vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 生成歌词
generated = generate_lyrics(
    model,
    dataset,
    start_string="我想要",
    length=200,
    temperature=0.8
)
print(generated)
```

## 温度参数说明

`temperature` 控制生成的随机性：

- **0.5-0.7**: 保守，更接近训练数据，重复性高
- **0.8-1.0**: 平衡，推荐使用
- **1.0-1.5**: 创新，更随机，可能出现不常见组合

## 训练输出示例

```
================================================================================
周杰伦歌词RNN生成模型
================================================================================
设备: cuda
RNN类型: LSTM
序列长度: 50
批大小: 64
...
================================================================================

加载数据...
数据集大小: 1234 个字符
词汇表大小: 156 个不同字符

创建模型...
模型参数数量: 234,567

开始训练...
Epoch [1/100] Loss: 3.2456 Perplexity: 25.67 Time: 2.34s
...
```

## 项目结构

```
.
├── rnn_lyrics_generation.py  # 主程序（单文件实现）
├── dataset/
│   └── jaychou_lyrics.txt    # 歌词数据
├── checkpoints/
│   └── rnn_lyrics_best.pth   # 保存的最佳模型
└── RNN_LYRICS_README.md      # 本说明文档
```

## 代码结构

单文件包含所有功能：

1. **数据预处理** (`LyricsDataset`)
   - 字符级tokenization
   - 词汇表构建
   - 序列生成

2. **模型定义** (`RNNLyricsModel`)
   - 嵌入层
   - LSTM/GRU层
   - 输出层

3. **训练循环** (`train_epoch`)
   - 梯度裁剪
   - 损失计算
   - 困惑度监控

4. **歌词生成** (`generate_lyrics`)
   - 温度采样
   - 自回归生成

## 进阶使用

### 切换到GRU

```python
# 在 main() 函数中修改
rnn_type = 'GRU'  # 从 'LSTM' 改为 'GRU'
```

### 调整模型大小

```python
# 更大的模型（需要更多显存）
hidden_dim = 512
num_layers = 3

# 更小的模型（训练更快）
hidden_dim = 128
num_layers = 1
```

### 增加训练数据

将更多歌词添加到 `dataset/jaychou_lyrics.txt` 可以提高生成质量。

## 常见问题

**Q: 显存不足怎么办？**
A: 减小 `batch_size` 或 `hidden_dim`

**Q: 生成的歌词质量不好？**
A:
- 增加训练数据量
- 增加训练轮数
- 调整温度参数
- 尝试不同的 `seq_length`

**Q: 训练时间太长？**
A:
- 使用GPU加速
- 减小模型大小
- 减少训练轮数

## 作者

KK266299
日期: 2025-11-20
