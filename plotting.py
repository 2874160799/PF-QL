import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque

KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
OSCILLATIONS_DETECTION_LENGTH = 3

class Plotting:
    def __init__(self, map_size=20, obstacle_ratio=0.2):
        """
        初始化Plotting类
        :param map_size: 地图大小（默认50×50）
        :param obstacle_ratio: 障碍物比例（默认20%）
        """
        self.map_size = map_size
        self.obstacle_ratio = obstacle_ratio
        self.obstacle = np.zeros((map_size, map_size), dtype=int)  # Initialize an empty grid
        self.start = (map_size-map_size,map_size-map_size)
        self.goal = (map_size-1,map_size-1)
        self._generate_obstacles()
        self.pamp = self.calc_pamp_filed()

        
    def _generate_obstacles(self):
        """生成障碍物"""
        np.random.seed(42)
        
        num_obstacles = int(self.map_size * self.map_size * self.obstacle_ratio)
        obstacle_indices = np.random.choice(self.map_size**2, num_obstacles, replace=False)
        obstacle_indices = [idx for idx in obstacle_indices if idx != 0 and idx != self.map_size**2 - 1]
        # 放置障碍物之前检查是否保证路径通畅
        while True:
            # 重置地图
            self.obstacle.fill(0)

            # 放置障碍物
            rows, cols = np.divmod(obstacle_indices, self.map_size)
            for row, col in zip(rows, cols):
                self.obstacle[row, col] = 1

            # 确保(0, 0)到(19, 19)有通路
            if self._bfs_path_exists((0, 0), (self.map_size - 1, self.map_size - 1)):
                break

            # 如果没有路径，则重新选择障碍物位置
            obstacle_indices = np.random.choice(self.map_size**2, num_obstacles, replace=False)
            obstacle_indices = [idx for idx in obstacle_indices if idx != 0 and idx != self.map_size**2 - 1]

        # 获取障碍物的位置
        rows, cols = np.divmod(obstacle_indices, self.map_size)
        self.obstacle_positions = list(zip(cols, rows))  # xy坐标
        # print(self.obstacle_positions)
        return self.obstacle_positions

    def plot_grid(self,path=None,save_path=None,dpi=300):
        
        fig,self.ax = plt.subplots(figsize=(self.map_size,self.map_size))
        self.ax.set_xlim(0,self.map_size)
        self.ax.set_ylim(0,self.map_size)
        self.ax.set_xticks(np.arange(0,self.map_size+1,1))
        self.ax.set_yticks(np.arange(0,self.map_size+1,1))
        self.ax.grid(True)
        self.ax.set_aspect('equal',adjustable='box')
        
        # 绘制障碍物
        for x,y in self.obstacle_positions:
            rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='gray', fill=True)
            self.ax.add_patch(rect)

        # 绘制起点
        start = patches.Circle((self.start[0]+0.5,self.start[1]+0.5),0.4,color='red',alpha=0.7)
        self.ax.add_patch(start)
        # 绘制终点
        goal = patches.Circle((self.goal[0]+0.5,self.goal[1]+0.5),0.4,color='blue',alpha=0.7)
        self.ax.add_patch(goal)
        
        # 绘制力场数字
        for x in range(self.map_size):
            for y in range(self.map_size): 
                self.ax.text(x+0.5,y+0.5,f'{self.pamp[x][y]:.2f}', ha='center', va='center',color='black', fontsize=12)
        
        
        
        # 绘制路径
        if path != None:
            for x,y in path:
                path_points = patches.Circle((x+0.5,y+0.5),0.4,color='green',alpha=0.7)
                self.ax.add_patch(path_points)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi)
        else:
            plt.show()
        plt.close()
        
        
    def _bfs_path_exists(self,start,end):
        """使用BFS算法检查从start到end是否有路径"""
        queue = deque([start])
        visited = set([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == end:
                return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size and (nx, ny) not in visited and self.obstacle[nx, ny] == 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False  
        

    def calc_pamp_filed(self):
        ox,oy = zip(*self.obstacle_positions)
        ox = [x+0.5 for x in ox]
        oy = [y+0.5 for y in oy]
        pamp = [[0.0 for i in range(self.map_size)] for i in range(self.map_size)]
        for ix in range(self.map_size):
            x = ix
            for iy in range(self.map_size):
                y = iy
                ug = self.calc_attractive_potential(x,y,self.goal[0],self.goal[1]) #引力
                # uo = self.calc_repulsive_potential(x,y,ox,oy,3) #斥力
                if (x,y) in self.obstacle_positions:
                    uf = 100
                else:
                    uf = ug 
                pamp[ix][iy] = uf
        return pamp
    
    def calc_attractive_potential(self,x,y,gx,gy):
        # return 65.4 - 0.5 * KP * np.hypot(x - gx, y - gy)
        return 0.5 * KP * np.hypot(x - gx, y - gy)
    

    def calc_repulsive_potential(self,x, y, ox, oy, rr):
        """ 计算斥力势能： 如果与最近障碍物的距离dq在机器人膨胀半径rr之内：1/2*ETA*(1/dq-1/rr)**2 否则：0.0 """
        # search nearest obstacle
        minid = -1
        dmin = float("inf")
        for i, _ in enumerate(ox):
            d = np.hypot(x - ox[i], y - oy[i])
            if dmin >= d:
                dmin = d
                minid = i

        # calc repulsive potential
        dq = np.hypot(x - ox[minid], y - oy[minid])

        if dq <= rr:
            if dq <= 0.1:
                dq = 0.1

            return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
        else:
            return 0.0 




# # 使用示例
# if __name__ == "__main__":
#     # 创建Plotting对象
#     plotter = Plotting(map_size=20, obstacle_ratio=0.2)
#     # 绘制地图并保存
#     plotter.plot_grid(save_path="/home/ubuntu/lyl/Q_learning/img/map.png",dpi=300)
#     pamp = plotter.calc_pamp_filed()
#     print(pamp)
#     # print(plotter.obstacle_positions)

