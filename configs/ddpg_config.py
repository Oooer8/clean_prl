import torch
from dataclasses import dataclass

@dataclass
class RepresentationConfig:
    type: str = "mlpnet"
    hidden_dims: tuple = (16, 16)  # 增加隐藏层大小
    output_dim: int = 6  # 增加表示层输出维度
    dropout: float = 0.1

@dataclass
class PolicyConfig:
    type: str = "mlp"
    hidden_dims: tuple = (64, 64)  # 增加隐藏层大小
    activation: str = "ReLU"
    output_activation: str = "tanh"  # 确保动作范围在[-1, 1]

@dataclass
class ValueConfig:
    type: str = "mlp"
    hidden_dims: tuple = (64, 64)  # 增加隐藏层大小
    activation: str = "ReLU"

@dataclass
class BufferConfig:
    type: str = "replay"
    capacity: int = 1000000  # 更大的缓冲区
    batch_size: int = 256    # 更大的批次大小

@dataclass
class AgentConfig:
    type: str = "ddpg"
    gamma: float = 0.99      # 折扣因子
    rep_lr: float = 1e-3     # 降低表示层学习率
    actor_lr: float = 1e-2   # 降低Actor学习率
    critic_lr: float = 3e-2  # 降低Critic学习率
    tau: float = 0.01       # 降低软更新系数，更加稳定
    update_every: int = 50   # 减少目标网络更新频率
    noise_scale: float = 1.0 # 噪声缩放因子
    noise_theta: float = 0.15 # OU噪声参数
    noise_sigma: float = 0.1  # OU噪声参数

@dataclass
class Config:
    representation: RepresentationConfig = RepresentationConfig()
    policy: PolicyConfig = PolicyConfig()
    value: ValueConfig = ValueConfig()
    buffer: BufferConfig = BufferConfig()
    agent: AgentConfig = AgentConfig()
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"