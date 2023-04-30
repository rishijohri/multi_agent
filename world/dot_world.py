import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class DotWorld(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'render-fps': 4}
    
    def __init__(self, 
                render_mode=None, 
                size:int=5, 
                agents:int =1, 
                error_reward:int=-2, 
                success_reward:int=10, 
                living_reward:int=-0.05, 
                img_mode:bool=False,
                episode_length:int=100):
        self.size  = size
        self.window_size = 500
        self.render_mode = render_mode
        self.agents = agents
        self.error_reward = error_reward
        self.success_reward = success_reward
        self.living_reward = living_reward
        self.episode_length = episode_length
        self.img_mode = img_mode
        self.steps = 0
        self.window = None
        self.clock = None
        self.observation_space = spaces.Dict({
            'agent': spaces.Sequence(spaces.Box(0, size-1, shape=(2,), dtype=np.int32)),
            # 'agent': [spaces.Box(0, size-1, shape=(2,), dtype=np.int32)]*agents,
            'target': spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
        })

        self.action_space = spaces.MultiDiscrete([5]*agents)
        self.single_action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0])
        }
    
    def _get_obs(self):
        if self.img_mode:
            return self._get_img_obs()
        return {
            'agent': self._agents_location,
            'target': self._target_location
        }
    
    def _get_img_obs(self):
        states = []
        for i in len(self._agents_location):
            state = self.render_frame_array(render_mode='rgb_array', agent=i)
            states.append(state)
        return states
    
    def _get_info(self):
        return {"distance": [np.linalg.norm(agent_location - self._target_location, ord=1) for agent_location in self._agents_location]}

    def _chk_in_list(self, x, lst):
        '''
        Return true if present, else false
        '''
        for i in lst:
            if np.array_equal(x, i):
                return True
        return False

    def reset(self, seed=None, options=None):
        '''
        Reset the environment. Seed is optional.
        '''
        super().reset(seed=seed)
        if seed:
            self._agents_location = np.rint([self.np_random.integers(0, self.size, size=2, dtype=int, seed=seed) for _ in range(self.agents)])
        else:
            self._agents_location = np.rint([self.np_random.integers(0, self.size, size=2, dtype=int) for _ in range(self.agents)])
        self._target_location = self._agents_location[0]

        while self._chk_in_list(self._target_location, self._agents_location):
            if seed:
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int, seed=seed)
            else:
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        obs = self._get_obs()
        info = self._get_info()
        self.steps = 0

        if self.render_mode == 'human':
            self._render_frame()
        
        return obs, info

    def step(self, action):
        '''
        Step the environment. Action is required. 
        '''
        self.steps += 1
        if self.steps > self.episode_length:
            # if episode is over
            truncated = True
            terminated = True
            reward = self.error_reward
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated,truncated, info
        terminated = False
        truncated = False
        new_agents_location = self._agents_location
        # action is an array of size agents and each element is an action. action can be 0,1,2,3,4 representing up, right, down, left, stay
        for i in range(self.agents):
            if self._chk_in_list(new_agents_location[i] + self._action_to_direction[action[i]], new_agents_location):
                # if the new location is already occupied
                new_agents_location[i] = new_agents_location[i]
            else:
                new_agents_location[i] = new_agents_location[i] + self._action_to_direction[action[i]]
                new_agents_location[i] = np.clip(new_agents_location[i], 0, self.size-1)

        if len(np.unique(new_agents_location, axis=0)) < self.agents:
            # if there is a collision
            observation = self._get_obs()
            info = self._get_info()
            reward = self.error_reward
            terminated = True 
            return observation, reward, terminated, truncated, info
        elif self._chk_in_list(self._target_location, new_agents_location):
            # if target is reached
            reward = self.success_reward
            terminated = True
            # print("Reached Reward")
        else:
            terminated = False
            reward = self.living_reward
        self._agents_location = new_agents_location
        observation = self._get_obs()
        info = self._get_info()
        # renders the environment
        if self.render_mode == 'human':
            self._render_frame()
        return observation, reward, terminated,truncated, info
    
    def render(self):
        '''
        Render the environment.
        '''
        if self.render_mode =='rgb_array':
            return self._render_frame()

    def _render_frame(self, render_mode=None, agent_id=None):
        '''
        Render the environment frame by frame
        '''
        render_mode = render_mode or self.render_mode
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # white
        pix_square_size = self.window_size / self.size

        # draw target
        
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(
            pix_square_size * self._target_location,
            (pix_square_size, pix_square_size),
        )
        )

        # draw agent
        for i, agent_location in enumerate(self._agents_location):
            if i==agent_id:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0),
                    (agent_location+0.5) * pix_square_size,
                    pix_square_size / 3,
                )
            else:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    (agent_location+0.5) * pix_square_size,
                    pix_square_size / 3,
                )
        # draw grid
        for x in range(self.size+1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1
            )
        
        if render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render-fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas))
            )
    
    def render_frame_array(self, agent_id=0):
        '''
        Render the environment frame by frame and return the frame as a numpy array
        '''
        array = self._render_frame(render_mode='rgb_array', agent_id=agent_id)
        return array

    def close(self):
        '''
        Close the environment. Call this method when you are done with the environment.
        '''
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
    


if __name__ == '__main__':
    env = DotWorld(render_mode='human', size=15, agents=3)
    state, _ = env.reset()
    print(type(state['agent'][0]), np.concatenate(state['agent']).ravel())
    while True:
        obs, rew, term, info = env.step(env.action_space.sample())
        state = np.concatenate([np.concatenate(obs['agent']), obs['target']])
        if term:
            print("Terminated with reward", rew, "and info", info)
            env.reset()
            