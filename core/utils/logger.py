import numpy as np
import time
from pathlib import Path
import torch

class Logger:
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.reward_file = open(self.log_dir / "rewards.csv", "w")
            self.reward_file.write("episode,reward,length,time\n")
    
    def log_episode(self, episode, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if self.log_dir:
            elapsed_time = time.time() - self.start_time
            self.reward_file.write(f"{episode},{reward},{length},{elapsed_time}\n")
            self.reward_file.flush()
    
    def get_stats(self, window=100):
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        
        if len(rewards) >= window:
            avg_reward = np.mean(rewards[-window:])
            avg_length = np.mean(lengths[-window:])
        else:
            avg_reward = np.mean(rewards) if len(rewards) > 0 else 0
            avg_length = np.mean(lengths) if len(lengths) > 0 else 0
        
        return {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "num_episodes": len(rewards)
        }
    
    def close(self):
        if self.log_dir:
            self.reward_file.close()