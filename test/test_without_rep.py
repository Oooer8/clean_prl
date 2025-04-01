import sys
import os
# Add the parent directory to the path so Python can find the clean_prl package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import gym
from configs.ddpg_without_rep_config import Config
from agents.ddpg_without_rep import DDPGAgentWithoutRep
from core.utils.logger import Logger
from envs.wrappers import DictActionWrapper, ObservationWrapper
from envs.adapters import DefaultAdapter
import matplotlib.pyplot as plt
import time
import argparse

def make_env(env_name, render=False):
    # Only set render_mode to 'human' if rendering is enabled
    render_mode = 'human' if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env = ObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = DictActionWrapper(env)
    return env

def plot_rewards(episode_rewards, window_size=10):
    """绘制奖励曲线，包括每集奖励和滑动平均"""
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    
    # 计算滑动平均
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', label=f'Moving Avg ({window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards (DDPG without Representation)')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('ddpg_without_rep_training_rewards.png')
    plt.close()

def set_seed(env, seed):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)
    if hasattr(env, 'seed'):
        env.seed(seed)
    elif hasattr(env, 'reset'):
        try:
            env.reset(seed=seed)
        except:
            pass

def main():
    args = parse_args()
    
    # 初始化
    config = Config()
    
    # 决定是否渲染环境
    should_render = args.render
    env = make_env(args.env, render=should_render)
    adapter = DefaultAdapter()
    
    # 设置随机种子
    set_seed(env, config.seed)
    
    # 获取维度
    state_dim = adapter.get_observation_space(env).shape[0]
    action_dim = adapter.get_action_space(env).shape[0]
    
    # 创建代理
    agent = DDPGAgentWithoutRep(state_dim, action_dim, config)
    logger = Logger()
    
    # 用于可视化的数据
    episode_rewards = []
    episode_losses = []
    training_start_time = time.time()
    
    print(f"Starting training on {args.env} with DDPG (without representation)")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Device: {config.device}, Batch size: {config.buffer.batch_size}")
    print(f"Rendering: {'Enabled' if should_render else 'Disabled'}")
    print("-" * 50)
    
    # 训练循环
    for episode in range(args.episodes):
        episode_start_time = time.time()
        
        # 重置环境和噪声过程
        try:
            # 尝试新版本的Gym API，reset返回(obs, info)
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                # 旧版本的Gym API，reset只返回obs
                obs = reset_result
                info = {}
        except Exception as e:
            print(f"Error during environment reset: {e}")
            # 尝试重新创建环境
            env.close()
            env = make_env(args.env, render=should_render)
            set_seed(env, config.seed)
            obs, info = env.reset()
        
        agent.noise.reset()
        
        state = adapter.adapt_observation(obs)
        total_reward = 0
        done = False
        steps = 0
        episode_loss = []
        
        while not done:
            # 选择带噪声的动作（探索）
            epsilon = max(0.1, 1.0 - episode/200)  # 衰减的探索率
            action = agent.act(state, epsilon=epsilon)
            
            # 执行环境步骤
            try:
                # 尝试新的Gym API（5个返回值）
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # 回退到旧的Gym API（4个返回值）
                next_obs, reward, done, info = env.step(action)
            
            next_state = adapter.adapt_observation(next_obs)
            
            # 存储转换
            agent.add_to_buffer(state, action, reward, next_state, done)
            
            # 学习
            if len(agent.buffer) > agent.config.buffer.batch_size:
                batch = agent.sample_from_buffer()
                loss = agent.learn(batch)
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # 计算平均损失
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        episode_rewards.append(total_reward)
        
        # 计算每集时间
        episode_time = time.time() - episode_start_time
        
        # 记录
        logger.log_episode(episode, total_reward, steps)
        
        # 每集结束后打印信息
        if episode % 10 == 0:
            stats = logger.get_stats()
            elapsed_time = time.time() - training_start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"Episode {episode}/{args.episodes} | " 
                  f"Reward: {total_reward:.1f} | "
                  f"Avg Reward: {stats['avg_reward']:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Steps: {steps} | "
                  f"Buffer: {len(agent.buffer)} | "
                  f"Epsilon: {epsilon:.2f} | "
                  f"Time: {episode_time:.2f}s | "
                  f"Total Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # 定期绘制奖励曲线
        if episode % args.plot_freq == 0 and episode > 0:
            plot_rewards(episode_rewards)
            print(f"Reward plot saved to ddpg_without_rep_training_rewards.png")
        
        # 定期保存模型
        if episode % args.save_freq == 0 and episode > 0:
            save_path = f"models/ddpg_without_rep_{args.env.replace('-', '_')}_{episode}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
            print(f"Model saved to {save_path}")
        
        # 根据render_freq决定是否需要切换渲染模式
        if args.render and ((episode + 1) % args.render_freq == 0):
            # 如果下一个episode需要改变渲染状态
            next_should_render = ((episode + 1) % args.render_freq == 0)
            if should_render != next_should_render:
                env.close()
                should_render = next_should_render
                env = make_env(args.env, render=should_render)
                set_seed(env, config.seed)
    
    # 训练结束，保存最终模型和图表
    final_save_path = f"models/ddpg_without_rep_{args.env.replace('-', '_')}_final.pt"
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    agent.save(final_save_path)
    plot_rewards(episode_rewards)
    
    # 打印训练总结
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"Training completed for {args.env} with DDPG (without representation)")
    print(f"Episodes: {args.episodes}")
    print(f"Final average reward: {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final model saved to: {final_save_path}")
    print(f"Reward plot saved to: ddpg_without_rep_training_rewards.png")
    print("="*50)
    
    env.close()
    logger.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPG agent without representation layer')
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0', help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--render_freq', type=int, default=40, help='Render every N episodes')
    parser.add_argument('--plot_freq', type=int, default=10, help='Frequency of plotting rewards')
    parser.add_argument('--save_freq', type=int, default=100, help='Frequency of saving model')
    parser.add_argument('--eval_freq', type=int, default=50, help='Frequency of evaluation')
    return parser.parse_args()

if __name__ == "__main__":
    main()