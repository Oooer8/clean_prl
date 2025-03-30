import torch
from dataclasses import dataclass

@dataclass
class RepresentationConfig:
    type: str = "mlpnet"
    hidden_dims: tuple = (64, 128, 256)
    output_dim: int = 256
    dropout: float = 0.1

@dataclass
class PolicyConfig:
    type: str = "mlp"
    hidden_dims: tuple = (256, 128)
    activation: str = "ReLU"
    output_activation: str = "tanh"

@dataclass
class ValueConfig:
    type: str = "mlp"
    hidden_dims: tuple = (256, 128)
    activation: str = "ReLU"
    exploration_noise: int = 0.2
    action_smoothing: int = 0.1
    sampling_size: int = 100
    use_gaussian: bool = True
    use_smoothing: bool = True

@dataclass
class BufferConfig:
    type: str = "replay"
    capacity: int = 100000
    batch_size: int = 128

@dataclass
class AgentConfig:
    type: str = "dqn"
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    update_every: int = 100

@dataclass
class Config:
    representation: RepresentationConfig = RepresentationConfig()
    policy: PolicyConfig = PolicyConfig()
    value: ValueConfig = ValueConfig()
    buffer: BufferConfig = BufferConfig()
    agent: AgentConfig = AgentConfig()
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
