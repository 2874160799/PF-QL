from env import Environment
from env import final_states
from plotting import Plotting
from agent_brain import QLearningTable
import time
# import originpro as op

def update():
    # Resulted list for the plotting Episodes via Steps
    steps = []
    # Summed costs for all episodes in resulted list
    all_costs = []
    
    #每个回合的奖励
    episode_rewards = []
    #上回合的奖励
    # last_episode_rewards = 0
    
    total_actions_per_episode = []  # 记录每个回合的所有动作
    for episode in range(200):
        # Initial Observation
        observation = env.reset() #将机器人放在（0，0）处并清空d字典

        # Updating number of Steps for each Episode
        i = 0 # 步数

        # Updating the cost for each episode
        cost = 0
        
        total_rewards = 0 # 记录本 episode 的总奖励
        # print(f"episode:{episode},e_greedy:{RL.epsilon}")
        print(episode)
        total_actions = []  # 存储当前回合的所有动作
        while True:
            
            # RL chooses action based on observation当前机器人的坐标位置
            action = RL.choose_action(str(observation)) #寻找动作的依据为以一定概率选择目前状态下动作值函数最大的动作，以一定概率随机选择（随机选择的目的是增加探索率）
            # print(action)
            # print(f"Actions for this round:{action}")
            total_actions.append(action)  # 记录每一步的选择动作
            # print(f"action:{action}")
            # RL takes an action and get the next observation and reward
            observation_, reward, done = env.step(action) #将该动作执行，得到奖励值，下个状态以及是否结束寻路标志

            # RL learns from this transition and calculating the cost
            cost += RL.learn(str(observation), action, reward, str(observation_))

            # 记录总奖励
            total_rewards += reward

            # Swapping the observations - current and next
            observation = observation_

            # Calculating number of Steps in the current Episode
            i += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                steps += [i]
                all_costs += [cost]
                episode_rewards.append(total_rewards)  # 保存当前回合的总奖励
                total_actions_per_episode.append(total_actions)  # 保存当前回合的所有动作
                # if total_rewards > last_episode_rewards:
                # RL.epsilon = 0.1-(episode/2000)*(0.1-0.001)
                # if episode >1000:
                    # RL.epsilon = 0.01
                break
    # # action_mapping = {0: '上', 1: '下', 2: '左', 3: '右'}
    # for episode, actions in enumerate(total_actions_per_episode):
    #     # action_directions = [action_mapping[action] for action in actions]  # 映射数字到方向
    #     print(f"Episode {episode + 1} actions: {actions}")   

    # Showing the final route
    env.final()
    # Showing the Q-table with values for each action
    RL.print_q_table()

    # Plotting the results
    RL.plot_results(steps, all_costs)

    # Plotting the rewards
    RL.plot_rewards(episode_rewards)
    
if __name__ =='__main__':
    
    plotting = Plotting(map_size=50, obstacle_ratio=0.2)
    obstacles = plotting._generate_obstacles()
    env = Environment(obstacles,plotting)
    RL = QLearningTable(plotting.map_size,
                        plotting.pamp,
                    actions=list(range(env.n_actions)),
                    learning_rate=0.8,#新信息对q值影响的权重
                    reward_decay=0.9,#智能体对未来奖励的重视程度
                    e_greedy=0.01) # ε：随机探索，1-ε：依赖q值
    env.start_time = time.time()
    update()
    path = list(final_states().values())
    # print(path)
    plotting.plot_grid(path,save_path="/home/ubuntu/lyl/Q_learning/img/map.png",dpi=300)
    if path:
        print("time:",env.end_time - env.start_time)
    else:
        print("NO PATH FOUND")   
    
    
    
    
    
    
    
    # print(RL.q_table.head(10))
    # print(RL.q_table_init)
    # print(len(plotting.pamp[0]))
    