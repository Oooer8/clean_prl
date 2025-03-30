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

def make_env(env_name):
    env = gym.make(env_name)
    env = ObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = DictActionWrapper(env)
    return env

def test_run(env_name="Pendulum-v1", num_episodes=1000):
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
    
    # Training loop
    for episode in range(num_episodes):
        obs = env.reset()
        state = adapter.adapt_observation(obs)
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select action
            action = agent.act(state, epsilon=max(0.1, 1 - episode/200))
            
            # Step environment
            next_obs, reward, done, _ = env.step(action)
            next_state = adapter.adapt_observation(next_obs)
            
            # Store transition
            agent.add_to_buffer(state, action, reward, next_state, done)
            
            # Learn
            if len(agent.buffer) > agent.config.buffer.batch_size:
                batch = agent.sample_from_buffer()
                loss = agent.learn(batch)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Logging
        logger.log_episode(episode, total_reward, steps)
        
        if episode % 10 == 0:
            stats = logger.get_stats()
            print(f"Episode {episode}: Reward={total_reward:.1f}, Avg Reward={stats['avg_reward']:.1f}")
    
    env.close()
    logger.close()

if __name__ == "__main__":
    test_run()
