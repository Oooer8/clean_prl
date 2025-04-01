import gym
import time

env = gym.make('Acrobot-v1', render_mode='human')
env.reset()

for _ in range(200):
    action = env.action_space.sample()
    _, _, done, _, _ = env.step(action)
    env.render()  # 显式调用渲染
    time.sleep(0.01)
    if done:
        env.reset()

env.close()