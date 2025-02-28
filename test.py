import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成模拟数据
np.random.seed(42)
x = np.linspace(0, 50, 100)
y1 = 1000 * np.log1p(x) + np.random.normal(0, 200, size=x.shape)
y2 = 800 * np.log1p(x) + np.random.normal(0, 150, size=x.shape)
y3 = 3000 * np.log1p(x / 5) + np.random.normal(0, 300, size=x.shape)
y4 = 2500 * np.log1p(x / 4) + np.random.normal(0, 250, size=x.shape)
human = np.full_like(x, 4500)

# 画图
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=y1, label="Async 1-step Q")
sns.lineplot(x=x, y=y2, label="Async SARSA")
sns.lineplot(x=x, y=y3, label="Async n-step Q")
sns.lineplot(x=x, y=y4, label="Async actor-critic")
sns.lineplot(x=x, y=human, label="Human tester", color="yellow")

# 添加标题和标签
plt.title("Slow car, no bots")
plt.xlabel("Training time (hours)")
plt.ylabel("Score")

plt.savefig("/home/ubuntu/lyl/Q_learning/img/test.png")

