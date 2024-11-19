import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# Nature 杂志风格的配色方案
colors = sns.color_palette("muted")

# 模拟数据
x = np.arange(1, 6)  # 超参数，5个数据点
y_data = [np.random.rand(5) * (i + 1) for i in range(4)]  # 生成4组不同的ASR数据

# 创建一个包含4个子图的图形
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

# 绘制每个子图
for i, ax in enumerate(axes):
    ax.plot(x, y_data[i], marker='o', color=colors[i], label=f'Experiment {i+1}')
    ax.set_title(f'Plot {i+1}')
    ax.set_xlabel('Hyperparameter')
    ax.set_ylabel('ASR')
    ax.set_xticks(x)
    ax.legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
