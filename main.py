import time # for sleep
from world import dot_world

env = dot_world.DotWorld(render_mode='human', size=50, agents=3)

env.reset()
done = False
for i in range(5):
    env.reset()
    while not done:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()

