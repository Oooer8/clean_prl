import torch
from dataclasses import dataclass

@dataclass
class PolicyConfig:
    type: str = "mlp"
    hidden_dims: tuple = (256, 256)
    activation: str = "ReLU"
    output_activation: str = "tanh"  # 确保动作范围在[-1, 1]

@dataclass
class ValueConfig:
    type: str = "mlp"
    hidden_dims: tuple = (256, 256)
    activation: str = "ReLU"

@dataclass
class BufferConfig:
    type: str = "replay"
    capacity: int = 1000000  # 更大的缓冲区
    batch_size: int = 256    # 更大的批次大小

@dataclass
class AgentConfig:
    type: str = "ddpg_without_rep"
    gamma: float = 0.99      # 折扣因子
    actor_lr: float = 1e-4   # Actor学习率
    critic_lr: float = 1e-3  # Critic学习率
    tau: float = 0.005       # 软更新系数
    update_every: int = 1    # 每步更新一次目标网络
    noise_scale: float = 1.0 # 噪声缩放因子
    noise_theta: float = 0.15 # OU噪声参数
    noise_sigma: float = 0.2  # OU噪声参数

@dataclass
class Config:
    policy: PolicyConfig = PolicyConfig()
    value: ValueConfig = ValueConfig()
    buffer: BufferConfig = BufferConfig()
    agent: AgentConfig = AgentConfig()
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"