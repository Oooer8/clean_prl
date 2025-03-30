import sys
import os
# Add the parent directory to the path so Python can find the clean_prl package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import gym
from configs.config import Config
from agents.dqn import DQNAgent
from core.utils.logger import Logger
from envs.wrappers import DictActionWrapper, ObservationWrapper
from envs.adapters import DefaultAdapter
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def make_env(env_name):
    env = gym.make(env_name)
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
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('training_rewards.png')
    plt.close()

def test_run(env_name="Pendulum-v1", num_episodes=1000, render_freq=50, plot_freq=10, save_freq=100):
    # Initialize
    config = Config()
    env = make_env(env_name)
    adapter = DefaultAdapter()
    
    # Get dimensions
    state_dim = adapter.get_observation_space(env).shape[0]
    action_dim = adapter.get_action_space(env).shape[0]
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config)
    logger = Logger()
    
    # 用于可视化的数据
    episode_rewards = []
    episode_losses = []
    training_start_time = time.time()
    
    print(f"Starting training on {env_name}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Device: {config.device}, Batch size: {config.buffer.batch_size}")
    print("-" * 50)
    
    # Training loop
    for episode in range(num_episodes):
        episode_start_time = time.time()
        
        # Reset environment
        obs, info = env.reset()

        state = adapter.adapt_observation(obs)
        total_reward = 0
        done = False
        steps = 0
        episode_loss = []
        
        # 是否渲染这一集
        should_render = (episode % render_freq == 0)
        
        while not done:
            # 渲染环境（如果需要）
            if should_render:
                try:
                    env.render()
                except:
                    # 某些环境可能不支持渲染
                    pass
            
            # Select action
            epsilon = max(0.1, 1 - episode/200)  # 衰减的探索率
            action = agent.act(state, epsilon=epsilon)
            
            # Step environment
            try:
                # Try the new Gym API (5 return values)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # Fall back to old Gym API (4 return values)
                next_obs, reward, done, info = env.step(action)
            
            next_state = adapter.adapt_observation(next_obs)
            
            # Store transition
            agent.add_to_buffer(state, action, reward, next_state, done)
            
            # Learn
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
        
        # Logging
        logger.log_episode(episode, total_reward, steps)
        
        # 每集结束后打印信息
        if episode % 10 == 0:
            stats = logger.get_stats()
            elapsed_time = time.time() - training_start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"Episode {episode}/{num_episodes} | " 
                  f"Reward: {total_reward:.1f} | "
                  f"Avg Reward: {stats['avg_reward']:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Steps: {steps} | "
                  f"Buffer: {len(agent.buffer)} | "
                  f"Epsilon: {epsilon:.2f} | "
                  f"Time: {episode_time:.2f}s | "
                  f"Total Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # 定期绘制奖励曲线
        if episode % plot_freq == 0 and episode > 0:
            plot_rewards(episode_rewards)
            print(f"Reward plot saved to training_rewards.png")
        
        # 定期保存模型
        if episode % save_freq == 0 and episode > 0:
            save_path = f"models/dqn_{env_name.replace('-', '_')}_{episode}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # 训练结束，保存最终模型和图表
    final_save_path = f"models/dqn_{env_name.replace('-', '_')}_final.pt"
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    agent.save(final_save_path)
    plot_rewards(episode_rewards)
    
    # 打印训练总结
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"Training completed for {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Final average reward: {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final model saved to: {final_save_path}")
    print(f"Reward plot saved to: training_rewards.png")
    print("="*50)
    
    env.close()
    logger.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_model_path': final_save_path
    }

if __name__ == "__main__":
    test_run()