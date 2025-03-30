import sys
import os
# Add the parent directory to the path so Python can find the clean_prl package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import gym
from configs.dqn_config import Config
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
    
        
        while not done:
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

        
        # 定期保存模型
        if episode % save_freq == 0 and episode > 0:
            save_path = f"models/dqn_{env_name.replace('-', '_')}_{episode}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":
    test_run()