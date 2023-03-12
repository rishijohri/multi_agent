import time # for sleep

from world import DotWorld

env = DotWorld(render_mode='human', size=50, agents=3)
stages = 3
env.reset()

for i in range(stages):
    env.reset()
    done = False
    print(f'Stage {i}')
    while not done:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()


