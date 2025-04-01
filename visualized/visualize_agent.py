import argparse
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from configs.ddpg_config import Config as DDPGConfig
from configs.ddpg_without_rep_config import Config as DDPGWithoutRepConfig
from agents.ddpg import DDPGAgent
from agents.ddpg_without_rep import DDPGAgentWithoutRep
from envs.wrappers import ObservationWrapper, DictActionWrapper
from envs.adapters import DefaultAdapter

"""
usage:
    python visualized/visualize_agent.py --model models/ddpg_Pendulum_v1_1200.pt  --env Pendulum-v1 --agent_type ddpg --episodes 5
    
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model (.pt file)")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--agent_type", type=str, default="ddpg", choices=["ddpg", "ddpg_without_rep"],
                        help="Type of agent (ddpg or ddpg_without_rep)")
    parser.add_argument("--render_mode", type=str, default="human", 
                        help="Render mode (human, rgb_array)")
    parser.add_argument("--delay", type=float, default=0.01, 
                        help="Delay between frames (seconds)")
    parser.add_argument("--record", action="store_true", help="Record video of agent performance")
    parser.add_argument("--output", type=str, default="videos", help="Directory to save videos")
    return parser.parse_args()

def make_env(env_name, render_mode="human"):
    """创建环境并应用适当的包装器"""
    try:
        # 尝试使用render_mode参数（新版gym）
        env = gym.make(env_name, render_mode=render_mode)
    except TypeError:
        # 如果失败，回退到不带render_mode的版本（旧版gym）
        env = gym.make(env_name)
    
    # 应用包装器
    env = ObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = DictActionWrapper(env)
    
    return env

def visualize(args):
    """可视化代理在环境中的表现"""
    print(f"Visualizing {args.agent_type} agent on {args.env} environment")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建环境
    env = make_env(args.env, args.render_mode)
    
    # 使用适配器获取状态和动作维度
    adapter = DefaultAdapter()
    state_dim = adapter.get_observation_space(env).shape[0]
    action_dim = adapter.get_action_space(env).shape[0]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # 创建代理
    if args.agent_type == "ddpg":
        config = DDPGConfig()
        agent = DDPGAgent(state_dim, action_dim, config)
    else:
        config = DDPGWithoutRepConfig()
        agent = DDPGAgentWithoutRep(state_dim, action_dim, config)
    
    # 加载模型
    print(f"Loading model from {args.model}...")
    try:
        agent.load(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 运行可视化
    total_rewards = []
    episode_lengths = []
    
    for episode in range(args.episodes):
        print(f"\nStarting episode {episode+1}/{args.episodes}")
        
        # 重置环境
        try:
            result = env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                # 新版gym API: (obs, info)
                state, _ = result
            else:
                # 旧版gym API: obs
                state = result
        except Exception as e:
            print(f"Error resetting environment: {e}")
            break
        
        # 适配观察
        state = adapter.adapt_observation(state)
        
        episode_reward = 0
        done = False
        truncated = False
        step = 0
        
        # 运行一个回合
        while not (done or truncated):
            # 获取动作
            action = agent.act(state, epsilon=0.0)  # 不添加探索噪声
            
            # 执行动作
            try:
                result = env.step(action)
                
                # 处理新旧gym API
                if len(result) == 5:  # 新版gym API
                    next_state, reward, done, truncated, info = result
                else:  # 旧版gym API
                    next_state, reward, done, info = result
                    truncated = False
            except Exception as e:
                print(f"Error during step: {e}")
                break
            
            # 适配下一个状态
            next_state = adapter.adapt_observation(next_state)
            
            # 更新累计奖励
            episode_reward += reward
            
            # 渲染
            try:
                env.render()
                time.sleep(args.delay)  # 添加延迟使可视化更容易观察
            except Exception as e:
                print(f"Error rendering: {e}")
            
            # 更新状态
            state = next_state
            step += 1
            
            # 打印进度
            if step % 100 == 0:
                print(f"  Step {step}, Current reward: {episode_reward:.2f}")
        
        # 记录回合结果
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
        print(f"Episode {episode+1} finished: Reward = {episode_reward:.2f}, Steps = {step}")
    
    # 关闭环境
    env.close()
    
    # 打印统计信息
    if total_rewards:
        print("\nResults Summary:")
        print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        print(f"Best episode: {np.max(total_rewards):.2f}")
        print(f"Worst episode: {np.min(total_rewards):.2f}")
    else:
        print("No episodes completed successfully.")
    
    # 绘制奖励图表
    if len(total_rewards) > 1:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(total_rewards) + 1), total_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'{args.agent_type.upper()} Agent on {args.env} - Visualization Results')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(f'{args.agent_type}_{args.env}_visualization.png')
        print(f"Results plot saved as {args.agent_type}_{args.env}_visualization.png")

if __name__ == "__main__":
    args = parse_args()
    visualize(args)