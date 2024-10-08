import math

import gym
import numpy as np
import matplotlib.pyplot as plt

# 环境初始化
env = gym.make('CartPole-v0')

# 参数设置
# 学习率, 用来控制新信息对旧信息的影响程度。0表示完全不学习，1表示完全学习
LEARNING_RATE = 0.1
# 折扣因子, 在0到1之间，衡量对未来奖励的重视程度
DISCOUNT = 0.95
# 迭代次数
EPISODES = 10000
# 显示间隔, 控制展示模拟环境的状态的频率
SHOW_INTERVAL = 500
# 更新间隔, 控制记录当前进程的频率
UPDATE_INTERVAL = 100

# 定义了ε-greedy策略中的探索参数，并设置了ε值的衰减方式
# 探索率ε
epsilon = 1
# 从第2次运行开始，ε值开始逐渐衰减
START_EPSILON_DECAYING = 2
# 结束衰减的时间点
# 在总运行次数的一半时，ε值停止衰减
END_EPSILON_DECAYING = EPISODES // 2


# 定义指数衰减函数
def exponential_decay(episode, start_epsilon, end_epsilon, decay_rate):
    return end_epsilon + (start_epsilon - end_epsilon) * math.exp(-decay_rate * episode)

# 创建离散化的状态区间 bins 和 Q 表
def create_bins_and_q_table():
	# 空间的上限 env.observation_space.high
	# [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
	# 空间的下限 env.observation_space.low
	# [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

	# 离散化状态区间的数量
	bins_nums = 20
	observation_space_size = len(env.observation_space.high)

	# Get the size of each bucket
	# 使用 np.linspace 创建每个维度的离散区间。每个区间包含 bins_nums 个等间距的点
	bins = [
		np.linspace(-4.8, 4.8, bins_nums),
		np.linspace(-4, 4, bins_nums),
		np.linspace(-.418, .418, bins_nums),
		np.linspace(-4, 4, bins_nums)
	]

	# 创建 Q 表，
	# 大小为 [bins_nums] * observation_space_size + [env.action_space.n]，
	# 其中 env.action_space.n 是动作空间的大小

	# 对于 CartPole 环境，
	# observation_space_size 为 4，
	# env.action_space.n 为 2，
	# 因此 Q 表的形状为 [20, 20, 20, 20, 2]
	q_table = (np.random.uniform
			  (low=-2, high=0,
			   size=([bins_nums] * observation_space_size + [env.action_space.n])))

	return bins, observation_space_size, q_table


# CartPole-v0的4个状态取值都是连续值，无法创建行数有限的表格。因此，需要先对状态进行离散化。
# 将环境的状态转换为离散状态索引，以便在 Q 表中查找对应的 Q 值
# state: 当前环境的状态
# bins: 每个维度的离散区间，是一个列表
# obsSpaceSize: 观察空间的维度大小，即状态向量的长度
def get_discrete_state(state, bins, observation_space_size):
	# 初始化离散状态索引
	state_index = []
	for i in range(observation_space_size):
		# -1 will turn bin into index
		# np.digitize 返回的索引是从 1 开始的，而我们需要从 0 开始的索引
		state_index.append(np.digitize(state[i], bins[i]) - 1)
	return tuple(state_index)


bins, observation_space_size, qTable = create_bins_and_q_table()


# 记录每次运行的得分
previous_nums = [0] * UPDATE_INTERVAL
# 记录绘图指标
# episode（回合数）, 平均得分, 最低得分, 最高得分
graph_elements = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES):
	# 重置环境并获取离散状态索引
	discreteState = get_discrete_state(env.reset(), bins, observation_space_size)
	# 检查环境是否已经完成，初始化为false
	done = False
	# how may movements cart has made
	# 初始化推车的移动次数
	step_num = 0

	while not done:
		# 在训练过程中周期性地展示模拟环境的状态
		if episode % SHOW_INTERVAL == 0:
			env.render()
		# 增加移动次数
		step_num += 1
		# 从Q表中根据epsilon-greedy策略选择动作
		# 若随机数大于epsilon，则选取当前状态下Q值最大的动作
		if np.random.random() > epsilon:
			action = np.argmax(qTable[discreteState])
		# 否则，随机选择一个动作进行探索
		else:
			action = np.random.randint(0, env.action_space.n)

		# 执行选定动作并获取新状态、奖励等信息
		newState, reward, done, _ = env.step(action)
		# 将新状态离散化
		newDiscreteState = get_discrete_state(newState, bins, observation_space_size)
		# 估计未来最优Q值
		maxFutureQ = np.max(qTable[newDiscreteState])
		# 当前Q值
		currentQ = qTable[discreteState + (action, )]

		# # 如果杆子倒下且未达到200步，给予-375的奖励
		if done and step_num < 200:
			reward = -375
		else:
			reward = 0

		# 计算新的Q值
		newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)

		# 更新Q表中的对应条目
		qTable[discreteState + (action, )] = newQ
		# 将当前状态更新为新状态
		discreteState = newDiscreteState

	# 记录本次运行得分
	previous_nums.append(step_num)

	# 在衰减范围内，每次运行都会衰减
	if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
		epsilon = exponential_decay(episode, 1, 0.01, 0.001)

	# 每隔一定次数的运行，记录一次绘图指标
	if episode % UPDATE_INTERVAL == 0:
		# 从名为previous_nums的列表中获取最后UPDATE_INTERVAL个元素，并将这些元素赋值给latest_episodes
		latest_episodes = previous_nums[-UPDATE_INTERVAL:]
		average_num = sum(latest_episodes) / len(latest_episodes)
		graph_elements['ep'].append(episode)
		graph_elements['avg'].append(average_num)
		graph_elements['min'].append(min(latest_episodes))
		graph_elements['max'].append(max(latest_episodes))
		print("Episode:", episode, "Average:", average_num, "Min:", min(latest_episodes), "Max:", max(latest_episodes))


env.close()

# Plot graph
# 绘制训练结果图
# x轴为回合数，y轴为平均得分、最低得分和最高得分
plt.plot(graph_elements['ep'], graph_elements['avg'], label="average rewards")
plt.plot(graph_elements['ep'], graph_elements['min'], label="min rewards")
plt.plot(graph_elements['ep'], graph_elements['max'], label="max rewards")
# 图例的位置为右下角
plt.legend(loc=4)
plt.show()