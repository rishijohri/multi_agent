import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class PredatorPreyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'render-fps': 4}
    
    def __init__(self,
                render_mode=None,
                size:int=10,
                vision:int=5,
                predator:int =3,
                prey:int =1,
                error_reward:int=-2,
                success_reward:int=10,
                living_reward:int=-0.05,
                img_mode:bool=False,
                episode_length:int=100):
        self.size  = size
        self.vision = vision
        self.window_size = 500
        self.render_mode = render_mode
        self.predator_num = predator
        self.prey_num = prey
        self.active_predator = [True for i in range(self.predator_num)]
        self.active_prey = [True for i in range(self.prey_num)]
        self.error_reward = error_reward
        self.success_reward = success_reward
        self.living_reward = living_reward
        self.episode_length = episode_length
        self.img_mode = img_mode
        self.steps = 0
        self.window = None
        self.clock = None
        self.render_scale = 1
        self.observation_space = spaces.Dict({
            'predator': spaces.Sequence(spaces.Box(0, size-1, shape=(2,), dtype=np.int32)),
            'prey': spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
        })
        total_actions = 5
        self.action_space_predator = spaces.MultiDiscrete([total_actions]*predator)
        self.action_space_prey = spaces.MultiDiscrete([total_actions]*prey)
        self.single_action_space = spaces.Discrete(total_actions)
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0])
        }
    
    def _get_obs(self):
        if self.img_mode:
            return self._get_np_arr_obs()
        return {
            'predator': self._predator_location,
            'prey': self._prey_location
        }
    
    def _get_np_arr_obs(self):
        predator_states = []
        prey_states = []
        for i in range(len(self._predator_location)):
            state = self._render_predator_frame(predator_id=i)
            predator_states.append(state)
        for i in range(len(self._prey_location)):
            state = self._render_prey_frame(prey_id=i)
            prey_states.append(state)
        return {
            "predator":predator_states, 
            "prey":prey_states
        }
    
    def _get_info(self):
        return {}
    
    def reset(self, *, seed: int=None, options=None):
        self._predator_location = np.random.randint(0, self.size, size=(self.predator_num, 2))
        self._prey_location = np.random.randint(0, self.size, size=(self.prey_num, 2))
        self.steps = 0
        self.active_pred = [True for i in range(self.predator_num)]
        self.active_prey = [True for i in range(self.prey_num)]
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_obs(), self._get_info()
    
    def _get_reward(self):
        # if any predator reaches prey, success. else, living reward
        rewards = [self.living_reward for i in range(self.predator_num)]
        for i in range(self.predator_num):
            if self._predator_location[i] in self._prey_location:
                rewards[i] = self.success_reward
        return rewards
    def _get_prey_reward(self):
        # if any predator reaches prey, success. else, living reward
        rewards = [self.success_reward for i in range(self.prey_num)]
        for i in range(self.prey_num):
            if self._prey_location[i] in self._predator_location:
                rewards[i] = 0
        return rewards
    
    def _is_done(self):
        # if all prey are gone or episode length is reached, done
        if self.steps >= self.episode_length:
            return True
        if np.sum(self.active_prey) == 0:
            return True
        return False

    def _is_valid_predator(self, location, index):
        # check if location is valid
        if location[0] < 0 or location[0] >= self.size or location[1] < 0 or location[1] >= self.size:
            return False
        if location in np.delete(self._predator_location, index, axis=0):
            return False
        return True
    
    def _is_valid_prey(self, location, index):
        # check if location is valid for prey of i'th index
        if location[0] < 0 or location[0] >= self.size or location[1] < 0 or location[1] >= self.size:
            return False
        if location in np.delete(self._prey_location, index, axis=0):
            return False
        return True
    
    def render(self):
        '''
        Render the environment.
        '''
        if self.render_mode =='rgb_array':
            return self._render_frame()

    def step(self, action_pred, action_prey):
        # action_pred is a list of actions for each predator
        # action_prey is a list of actions for each prey
        # action is a number from 0 to 3
        # 0: up, 1: right, 2: down, 3: left
        # if in the new locations after actions two predator overlap each other, the action will not take place
        # if in the new locations after actions a predator overlaps a prey, the prey will be eaten and predator will continue
        # if in the new locations after actions a predator overlaps a wall or is on the edge, the predator will not move
        # if in the new locations after actions a prey overlaps a wall or is on the edge, the prey will not move
        # if in the new locations after actions a prey overlaps a predator, the predator will eat the prey and predator will continue

        if self._is_done():
            raise RuntimeError("Episode is done")
        self.steps += 1
        # move predator
        for i in range(self.predator_num):
            if self.active_predator[i] == False:
                continue
            if i < len(action_pred):
                
                action = action_pred[i]
            else:
                action = self.single_action_space.sample()
            new_location = self._predator_location[i] + self._action_to_direction[action]
            if self._is_valid_predator(new_location, i):
                self._predator_location[i] = new_location

        # move prey
        for i in range(self.prey_num):
            if self.active_prey[i] == False:
                continue
            if i < len(action_prey):
                action = action_prey[i]
            else:
                action = self.single_action_space.sample()
            
            new_location = self._prey_location[i] + self._action_to_direction[action]
            if self._is_valid_prey(new_location, i):
                self._prey_location[i] = new_location

        # check if any predator reaches prey and give reward
        pred_reward = self._get_reward()
        prey_reward = self._get_prey_reward()
        for i in range(self.predator_num):
            for j in range (self.prey_num):
                if np.all(self._predator_location[i] == self._prey_location[j]):
                    print("EATEN !!!")
                    self.active_prey[j] = False
        
        done = self._is_done()
        reward = {
            'predator': pred_reward,
            'prey': prey_reward
        }
        if self.render_mode == 'human':
            self._render_frame()

        return self._get_obs(), reward, done, self._get_info()
        
    def _render_predator_frame(self, predator_id:int=None):
        # the predator with predator_id will be in the center of the frame and the frame will be of size vision x vision
        # the predator with predator_id will be green
        # if predator_id is None, the function ends
        # if the predator is on the edge of the frame, the cells outside the grid will be white
        # the predator is blue
        # the prey is red

        if predator_id==None:
            return
        frame = np.zeros((3, self.vision, self.vision), dtype=np.uint8)
        # draw predator
        pred_loc = self._predator_location[predator_id]
        min_pred_loc = pred_loc - np.array([self.vision//2, self.vision//2])
        max_pred_loc = pred_loc + np.array([self.vision//2, self.vision//2])

        # add predator to centre of frame
        frame[1, self.vision//2, self.vision//2] = self.render_scale 
        # for each predator or prey within min and max it will be added in the frame
        for i in range(self.predator_num):
            if (min_pred_loc[0] <= self._predator_location[i][0] <= max_pred_loc[0] 
            and 
            min_pred_loc[1] <= self._predator_location[i][1] <= max_pred_loc[1]):
                frame[2, self._predator_location[i][0]-min_pred_loc[0], self._predator_location[i][1]-min_pred_loc[1]] = self.render_scale 
        
        for i in range(self.prey_num):
            if (min_pred_loc[0] <= self._prey_location[i][0] <= max_pred_loc[0] 
            and 
            min_pred_loc[1] <= self._prey_location[i][1] <= max_pred_loc[1]):
                frame[0, self._prey_location[i][0]-min_pred_loc[0], self._prey_location[i][1]-min_pred_loc[1]] = self.render_scale 
                
        # create white for cells outside grid
        if min_pred_loc[0] < 0:
            frame[:, :abs(min_pred_loc[0]), :] = self.render_scale 
        if max_pred_loc[0] >= self.size:
            frame[:, -(max_pred_loc[0]-self.size+1):, :] = self.render_scale 
        if min_pred_loc[1] < 0:
            frame[:, :, :abs(min_pred_loc[1])] = self.render_scale 
        if max_pred_loc[1] >= self.size:
            frame[:, :, -(max_pred_loc[1]-self.size+1):] = self.render_scale 
        
        return frame

    def _render_prey_frame(self, prey_id:int=None):
        # the prey will be in the centre of the frame and the frame will be of size vision x vision
        # the prey will be red
        # the prey with prey_id will be green
        # the predator is blue
        # if prey_id is None, the function ends
        # if the prey is on the edge of the frame, the cells outside the grid will be white

        if prey_id==None:
            return
        frame = np.zeros((3, self.vision, self.vision), dtype=np.uint8)
        # draw prey
        prey_loc = self._prey_location[prey_id]
        min_prey_loc = prey_loc - np.array([self.vision//2, self.vision//2])
        max_prey_loc = prey_loc + np.array([self.vision//2, self.vision//2])

        # add prey to centre of frame
        frame[1, self.vision//2, self.vision//2] = self.render_scale 
        # for each predator or prey within min and max it will be added in the frame
        for i in range(self.predator_num):
            if (min_prey_loc[0] <= self._predator_location[i][0] <= max_prey_loc[0] 
            and 
            min_prey_loc[1] <= self._predator_location[i][1] <= max_prey_loc[1]):
                frame[2, self._predator_location[i][0]-min_prey_loc[0], self._predator_location[i][1]-min_prey_loc[1]] = self.render_scale 
        
        for i in range(self.prey_num):
            if (min_prey_loc[0] <= self._prey_location[i][0] <= max_prey_loc[0] 
            and 
            min_prey_loc[1] <= self._prey_location[i][1] <= max_prey_loc[1]):
                frame[0, self._prey_location[i][0]-min_prey_loc[0], self._prey_location[i][1]-min_prey_loc[1]] = self.render_scale 
        
        # create white for cells outside grid
        if min_prey_loc[0] < 0:
            frame[:, :abs(min_prey_loc[0]), :] = self.render_scale 
        if max_prey_loc[0] >= self.size:
            frame[:, -(max_prey_loc[0]-self.size+1):, :] = self.render_scale 
        if min_prey_loc[1] < 0:
            frame[:, :, :abs(min_prey_loc[1])] = self.render_scale 
        if max_prey_loc[1] >= self.size:
            frame[:, :, -(max_prey_loc[1]-self.size+1):] = self.render_scale 
        
        return frame
    
    def _render_frame(self):
        # Renders the entire environment frame by frame
        # For visualisation purposes only

        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pixel_size = self.window_size // self.size

        # draw grid
        for i in range(self.size):
            pygame.draw.line(canvas, (0, 0, 0), (0, i*pixel_size), (self.window_size, i*pixel_size))
            pygame.draw.line(canvas, (0, 0, 0), (i*pixel_size, 0), (i*pixel_size, self.window_size))
        
        # draw prey as rectangle
        for i in range(self.prey_num):
            if self.active_prey[i]:
                pygame.draw.rect(canvas, (255, 0, 0), (self._prey_location[i][1]*pixel_size, self._prey_location[i][0]*pixel_size, pixel_size, pixel_size))

        # draw predator as circle
        for i in range(self.predator_num):
            if self.active_predator[i]:
                pygame.draw.circle(canvas, (0, 0, 255), (self._predator_location[i][1]*pixel_size+pixel_size//2, self._predator_location[i][0]*pixel_size+pixel_size//2), pixel_size//2)
        
        
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render-fps'])
        else:
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None


if __name__=='__main__':
    env = PredatorPreyEnv(
        render_mode='human',
        img_mode=True
    )
    env.reset()
    print(env.action_space_predator.sample())
    for i in range(100):
        pred_action = env.action_space_predator.sample()
        prey_action = env.action_space_prey.sample()
        print(pred_action, prey_action)
        obs, rew, done, info = env.step(pred_action, prey_action)
        print(obs["predator"][0][0, :, :], done)
    env.close()