import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import final_states
from env import Environment
from plotting import Plotting
import ast

class QLearningTable:
    def __init__(self,map_size,pamp,actions,learning_rate = 0.01,reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.map_size = map_size
    
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_init = self.initialize_q_table(pamp)
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.existed_coord_state = []
        self.obs_state = []
        self.all_state = []
    # Function for choosing the action for the agent
    def choose_action(self, observation):
        #转换为元组类型
        # observation = ast.literal_eval(observation)
        # Checking if the state exists in the table
        self.check_state_exist(observation)
        # Selection of the action - 90 % according to the epsilon == 0.9
        # Choosing the best action
        if np.random.uniform() < self.epsilon: #0
            # Choosing random action - left 10 % for choosing randomly
            action = np.random.choice(self.actions) #随机选择
        else:
            if self.shock_detection(observation) == False: # 没有遇到走重复路的方式
                    
                # print(self.q_table)
                # print(observation)
                state_action = self.q_table.loc[observation, :] #提取当前状态下所有动作的价值函数

                state_action = state_action.reindex(np.random.permutation(state_action.index))#打乱顺序，避免每次选择的动作都为序号偏前的动作
                action = state_action.idxmax()
            else:# 遇到了走过的路,随机选一个非障碍物的值
                state_action = self.q_table.loc[observation, :]
                state_action = state_action.reindex(np.random.permutation(state_action.index))  # 打乱顺序，防止偏向序号小的动作
                state_action = state_action[state_action != -20]
                action = np.random.choice(state_action.index)
        return action
    
    # Function for learning and updating Q-table with new knowledge
    def learn(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table 如果不在Q-table中则将其加入到Q-table中
        self.check_state_exist(next_state) 

        # Current state in the current position 
        q_predict = self.q_table.loc[state, action]  #预测的Q值,即目前Q_table内存储的Q值


        # Checking if the next state is free or it is obstacle or goal
        # 这句判断永远成立
        # if next_state != 'goal' or next_state != 'obstacle':
        if next_state not in ['goal', 'obstacle']:
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()  #实际最大值 由动作奖励以及下一状态的最大Q值×折损率组成
                        #当前奖励和未来的最大q值
        else:
            q_target = reward

        # Updating Q-table with new knowledge
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict) #更新Q值
        # print(self.q_table)
        td_error = q_target - q_predict
        return abs(td_error)
        # return self.q_table.loc[state, action]
        
    #确保进入新的状态时，在q表中为该状态添加一行
    def check_state_exist(self, state):
        # print(f"state:{state}")
        if state not in ['goal', 'obstacle']:# 当state不是goal或obstacle,为一个普通的坐标时
            if state not in self.existed_coord_state:
                self.existed_coord_state.append(state)
                # print("state not in ['goal', 'obstacle']")
                state_tuple = ast.literal_eval(state)
                
                # print(f"state_tuple:{state_tuple}")#也要加新的索引，但这个索引直接找到
                                                #self.q_table_init里相同内容即可
                row_values = self.q_table_init.loc[state_tuple].values  # 获取该行的值
                # 创建新行，并将获取的值赋给它
                existed_row = pd.Series(
                    row_values,
                    index=self.q_table.columns,
                    name=state,
                )
                self.q_table = pd.concat([self.q_table, existed_row.to_frame().T])
                # print(self.q_table)
        else:#当state为goal或obstacle时，需要新增新的索引行
            # print("state in ['goal', 'obstacle']")
            if state not in self.obs_state:
                self.obs_state.append(state)
                new_row = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
                # 使用 pd.concat 来替代 append
                self.q_table = pd.concat([self.q_table, new_row.to_frame().T])
                # print(self.q_table)
            
    def shock_detection(self,state):
        if state not in self.all_state:
            state = self.all_state.append(state)
            return False
        else:    
            return True   

    
    

    
    # Printing the Q-table with states
    def print_q_table(self):
        # Getting the coordinates of final route from env.py
        e = final_states()

        # Comparing the indexes with coordinates and writing in the new Q-table values
        for i in range(len(e)):
            state = str(e[i])  # state = '[5.0, 40.0]'
            # Going through all indexes and checking
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)
        
        
        
    def initialize_q_table(self, pamp):
        q_table = pd.DataFrame(index=pd.MultiIndex.from_product([range(self.map_size), range(self.map_size)]), columns=self.actions, dtype=np.float64)
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                q_values = {}
                
                # 上方向的Q值：pamp[x][y+1]，如果y+1超出范围，则设置为-100
                if y < self.map_size - 1:  
                    q_values[0] = -pamp[x][y+1]  # 上
                else:
                    q_values[0] = -200  # 不允许向上移动

                # 下方向的Q值：pamp[x][y-1]，如果y-1超出范围，则设置为-100
                if y > 0:  
                    q_values[1] = -pamp[x][y-1]  # 下
                else:
                    q_values[1] = -200  # 不允许向下移动

                # 左方向的Q值：pamp[x-1][y]，如果x-1超出范围，则设置为-100
                if x > 0:  
                    q_values[2] = -pamp[x-1][y]  # 左
                else:
                    q_values[2] = -200  # 不允许向左移动

                # 右方向的Q值：pamp[x+1][y]，如果x+1超出范围，则设置为-100
                if x < self.map_size - 1:  
                    q_values[3] = -pamp[x+1][y]  # 右
                else:
                    q_values[3] = -200  # 不允许向右移动

                # 将q_values应用到Q表中
                q_table.loc[(x, y), :] = pd.Series(q_values)

        return q_table * 0.1  # 调整Q值的范围
        




            
        
                
                
# Plotting the results for the number of steps
    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        # 保存子图
        # f.savefig('/home/ubuntu/catkin_hector_ws/qlearning/Reinforcement-Learning_Path-Planning/img/results(NO_APF_2000).png')
        f.savefig('/home/ubuntu/lyl/Q_learning/img/results.png')
        
        plt.close(f)
        
        
        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        # plt.savefig('/home/ubuntu/catkin_hector_ws/qlearning/Reinforcement-Learning_Path-Planning/img/steps(NO_APF_2000).png')
        plt.savefig('/home/ubuntu/lyl/Q_learning/img/steps.png')
        
        plt.close()
        steps_episodes = np.arange(len(steps))
        steps_data = pd.DataFrame({"Episodes":steps_episodes,"steps":steps})
        steps_csv_path = "/home/ubuntu/lyl/Q_learning/data/steps_data.csv"
        steps_data.to_csv(steps_csv_path,index=False)
        
        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        # plt.savefig('/home/ubuntu/catkin_hector_ws/qlearning/Reinforcement-Learning_Path-Planning/img/costs(NO_APF_2000).png')
        plt.savefig('/home/ubuntu/lyl/Q_learning/img/costs.png')
        
        plt.close()

        # Showing the plots
        # plt.show()
        episodes = np.arange(len(cost))
        data = pd.DataFrame({"Episodes":episodes,"cost":cost})
        csv_path = "/home/ubuntu/lyl/Q_learning/data/cost_data.csv"
        data.to_csv(csv_path,index=False)
    
    def plot_rewards(self,episode_rewards):
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label="Total Reward per Episode", color='b')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Trend Over Episodes")
        plt.legend()
        plt.grid()
        # plt.savefig('/home/ubuntu/catkin_hector_ws/qlearning/Reinforcement-Learning_Path-Planning/img/rewards(NO_APF_2000).png')
        plt.savefig('/home/ubuntu/lyl/Q_learning/img/rewards.png')
        
        plt.close()
        
        episodes = np.arange(len(episode_rewards))
        data = pd.DataFrame({"Episodes":episodes,"reward":episode_rewards})
        csv_path = "/home/ubuntu/lyl/Q_learning/data/reward_data.csv"
        data.to_csv(csv_path,index=False)
        
# if __name__=='__main__':
#     plotting = Plotting(map_size=30, obstacle_ratio=0.2)
#     obstacles = plotting._generate_obstacles()
#     env = Environment(obstacles,plotting)
#     RL = QLearningTable(plotting.map_size,
#                     plotting.pamp,
#                     actions=list(range(env.n_actions)),
#                     learning_rate=0.8,#新信息对q值影响的权重
#                     reward_decay=0.9,#智能体对未来奖励的重视程度
#                     e_greedy=0.1) # ε：随机探索，1-ε：依赖q值
#     qtable = RL.initialize_q_table(plotting.pamp)
#     # print(plotting.pamp)
#     print(qtable)
#     path = None
#     plotting.plot_grid(path,save_path="/home/ubuntu/lyl/Q_learning/img/map.png",dpi=300)
    
    
    
    