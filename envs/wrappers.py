import gym
import numpy as np
from typing import Dict, Any, Tuple, Union

class DictActionWrapper(gym.Wrapper):
    """Convert dict action space to flat vector"""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict), "Env must have Dict action space"
        
        self.action_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(self._get_action_dim(),),
            dtype=np.float32
        )
        
    def _get_action_dim(self):
        return sum([space.shape[0] for space in self.env.action_space.spaces.values()])
    
    def step(self, action):
        # Convert flat action back to dict
        action_dict = {}
        idx = 0
        for key, space in self.env.action_space.spaces.items():
            dim = space.shape[0]
            action_dict[key] = action[idx:idx+dim]
            idx += dim
        
        # Handle both old and new Gym API formats
        result = self.env.step(action_dict)
        
        # Check if we're dealing with new API (5 values) or old API (4 values)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, info
        else:
            obs, reward, done, info = result
            return obs, reward, done, info

class ObservationWrapper(gym.ObservationWrapper):
    """Ensure observation is dict with 'state' key"""
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                'state': env.observation_space
            })
    
    def observation(self, observation):
        if isinstance(observation, dict):
            return observation
        return {'state': observation}

    
    def reset(self, **kwargs):
        # Handle both old and new Gym API formats for reset
        try:
            # New Gym API might return (obs, info)
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
                return self.observation(obs), info
            else:
                # Old API just returns obs
                return self.observation(result)
        except TypeError:
            # If kwargs are not supported (old API)
            return self.observation(self.env.reset())

class PhysicsWrapper(gym.Wrapper):
    """Add physics-related observations"""
    def __init__(self, env):
        super().__init__(env)
        
        # Extend observation space
        original_space = self.env.observation_space
        new_spaces = {
            **original_space.spaces,
            'physics': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        }
        self.observation_space = gym.spaces.Dict(new_spaces)
    
    def step(self, action):
        # Handle both old and new Gym API formats
        result = self.env.step(action)
        
        # Check if we're dealing with new API (5 values) or old API (4 values)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            # Add physics info
            obs['physics'] = np.zeros(6)  # Replace with actual physics data
            return obs, reward, terminated, truncated, info
        else:
            obs, reward, done, info = result
            # Add physics info
            obs['physics'] = np.zeros(6)  # Replace with actual physics data
            return obs, reward, done, info
    
    def reset(self, **kwargs):
        # Handle both old and new Gym API formats for reset
        try:
            # New Gym API might return (obs, info)
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
                # Add physics info
                obs['physics'] = np.zeros(6)  # Replace with actual physics data
                return obs, info
            else:
                # Old API just returns obs
                obs = result
                # Add physics info
                obs['physics'] = np.zeros(6)  # Replace with actual physics data
                return obs
        except TypeError:
            # If kwargs are not supported (old API)
            obs = self.env.reset()
            # Add physics info
            obs['physics'] = np.zeros(6)  # Replace with actual physics data
            return obs