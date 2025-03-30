import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any

class EnvAdapter:
    """Base class for environment adapters"""
    @staticmethod
    def get_observation_space(env: gym.Env) -> spaces.Space:
        """Get modified observation space"""
        raise NotImplementedError
    
    @staticmethod
    def get_action_space(env: gym.Env) -> spaces.Space:
        """Get modified action space"""
        raise NotImplementedError
    
    @staticmethod
    def adapt_observation(obs: Any) -> np.ndarray:
        """Convert raw observation to algorithm-compatible format"""
        raise NotImplementedError
    
    @staticmethod
    def adapt_action(action: np.ndarray) -> Any:
        """Convert algorithm action to env-compatible format"""
        raise NotImplementedError

class DefaultAdapter(EnvAdapter):
    """Default adapter for simple Box spaces"""
    @staticmethod
    def get_observation_space(env):
        if isinstance(env.observation_space, spaces.Dict):
            return spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(sum([space.shape[0] for space in env.observation_space.spaces.values()]),),
                dtype=np.float32
            )
        return env.observation_space
    
    @staticmethod
    def get_action_space(env):
        if isinstance(env.action_space, spaces.Dict):
            return spaces.Box(
                low=-1, high=1,
                shape=(sum([space.shape[0] for space in env.action_space.spaces.values()]),),
                dtype=np.float32
            )
        return env.action_space
    
    @staticmethod
    def adapt_observation(obs):
        if isinstance(obs, dict):
            flattened_values = []
            for v in obs.values():
                if hasattr(v, 'flatten'):
                    flattened_values.append(v.flatten())
                elif isinstance(v, (list, tuple)):
                    flattened_values.append(np.array(v, dtype=np.float32).flatten())
                else:
                    flattened_values.append(np.array([v], dtype=np.float32))
            return np.concatenate(flattened_values)
        return obs
    
    @staticmethod
    def adapt_action(action):
        return action