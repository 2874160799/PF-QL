import numpy as np  
import math 
from plotting import Plotting
import time

# Global variable for dictionary with coordinates for the final route
a = {}


class Environment:
    def __init__(self,obstacles,plotting):
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.start = (plotting.map_size-plotting.map_size,plotting.map_size-plotting.map_size)
        self.goal = (plotting.map_size-1,plotting.map_size-1)
        self.coords = (plotting.map_size-plotting.map_size,plotting.map_size-plotting.map_size)
        self.obstacles = obstacles
        self.x_range = plotting.map_size 
        self.y_range = plotting.map_size
        self.start_time = 0.0
        self.end_time = 0.0
        self.first_flag = True
        self.map_size = plotting.map_size
        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}
        # Key for the dictionaries
        self.i = 0
        # Showing the steps for longest found route
        self.longest = 0
        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for the shortest route
        self.shortest = 0
        # print(self.obstacles)

    def reset(self):
        """Function to reset the environment and start new Episode"""
        # Updating agent
        self.coords=self.start #将坐标置为起点
        # Clearing the dictionary and the i
        self.d = {}
        self.i = 0
        # Return observation
        return self.coords

    def step(self,action):
        """Function to get the next observation and reward by doing next step"""
        # Current state of the agent
        state = self.coords
        base_action = [0,0]
        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1]<self.y_range - 1:
                base_action[1]+=1 
        # Action 'down'
        elif action == 1:
            if state[1]>=1:
                base_action[1]-=1 
        # Action left
        elif action == 2:
            if state[0]>=1:
                base_action[0]-=1 
        # Action right
        elif action == 3:
            if state[0]<self.x_range - 1:
                base_action[0]+=1 
                
        # Moving the agent according to the action
        self.coords=(self.coords[0]+base_action[0],self.coords[1]+base_action[1])
        #self.canvas_widget.move(self.agent, base_action[0], base_action[1])
        # print(f"Agent moved to: {self.coords}")
        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.coords
        
        # Updating next state
        next_state = self.d[self.i]
        
        # Updating key for the dictionary
        self.i += 1
        
        # Calculating the reward for the agent
        if next_state == self.goal:
            if self.first_flag == True:
                self.end_time = time.time()
                self.first_flag = False
            reward = 30
            done = True
            next_state = 'goal'

            # Filling the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) < len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

        elif self.is_collision(next_state):
            reward = -100
            done = True
            next_state = 'obstacle'

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0
        
        # elif self.is_visited(next_state):
        #     reward = -5
        #     done = False
            

        else:
            # pamp = self.calc_pamp_filed()#每次都要重新计算势场，导致训练速度很慢
            # reward = - (pamp[state[0]][state[1]] * 0.1)
            
            reward = -1
            done = False

        return next_state, reward, done        
        
    # Function to show the found route
    def final(self):
        # Deleting the agent at the end
       # self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        # origin = np.array([20, 20])
        # self.initial_point = self.canvas_widget.create_oval(
        #     origin[0] - 5, origin[1] - 5,
        #     origin[0] + 5, origin[1] + 5,
        #     fill='blue', outline='blue')

        # Filling the route
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            print(self.f[j])
            #self.track = self.canvas_widget.create_oval(
                #self.f[j][0] + origin[0] - 5, self.f[j][1] + origin[0] - 5,
                #self.f[j][0] + origin[0] + 5, self.f[j][1] + origin[0] + 5,
                #fill='blue', outline='blue')
            # Writing the final route in the global variable a
            a[j] = self.f[j]        
    
    def is_collision(self,state):
        """
        检测智能体是否与障碍物发生碰撞
        :param state: 智能体的当前坐标 (x, y)
        :return: 如果发生碰撞返回 True，否则返回 False
        """
        if state in self.obstacles or not(0<=state[0]<=self.x_range and 0<=state[1]<=self.y_range):
            return True
        else:
            return False
        
        

                

def final_states():
    return a
        


        
# if __name__=='__main__':
#     plotting = Plotting()
#     obstacles = plotting._generate_obstacles()
#     env = Environment(obstacles,plotting)
#     # print(env.obstacles)
#     pamp = env.calc_pamp_filed()
#     print(pamp)
#     # print(len(pamp[1]))

        