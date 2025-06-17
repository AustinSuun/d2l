"""
r2l.py - 动手学强化学习的包，这是动手学强化学习的练习代码，包含其中使用的部分代码

Author: github@AustinSuun
Data: 2025-05-24
"""

__version__ = "0.1"

# 核心库
import copy
import random
import collections
import itertools

# 三方库
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.optim.adam
from tqdm import tqdm  # 显示循环进度条
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import truncnorm

"""
代码从第四章开始
"""
# 第四章 动态规划算法


class CliffWalkingEnv:
    """悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 状态转移矩阵 P[state][action] = [P, next_state, reward, done]
        self.P = self.creatP()

    def creatP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.ncol * self.nrow)]
        # 4种动作，change[0:3]分别是上、下、左、右
        # 坐标原点(0,0)，定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置悬崖或者目标状态，因为无法交互，任何动作奖励都是0
                    if i == self.nrow - 1 and j > 0:  # 最后一行除了起点，包括悬崖和终点
                        # 在悬崖或者终点，选择每个方向收益都是 0，done是结束标志
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue

                    # 其他位置| 比下界大，比上界小
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False  # 没有结束游戏
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


## 策略迭代算法
class PolicyIteration:
    """策略迭代算法"""

    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值都是 0
        self.pi = [
            [0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)
        ]  # 初始化随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_envaluation(self):
        """策略评估"""
        cnt = 1  # 计数器
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 计算状态 s 下的所有 Q(s,a)的价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(self.v[s] - new_v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break  # 满足收敛条件，推出评估迭代
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):
        """策略提升"""
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 查看有几个动作是最大的Q值

            # 让这些动作平摊概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):
        """策略迭代"""
        while 1:
            self.policy_envaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 保持整齐，输出6个字符
            print("%6.6s" % ("%.3f" % agent.v[i * agent.env.ncol + j]), end=" ")
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态，如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print("⚠️⚠️⚠️⚠️", end=" ")
            elif (i * agent.env.ncol + j) in end:  # 目标状态|终点
                print("✅✅✅✅", end=" ")
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ""
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else "🅾️"

                print(pi_str, end=" ")
        print()


## 价值迭代算法


class ValueIteration:
    """价值迭代算法"""

    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma  # 折扣因子
        self.v = [0] * self.env.ncol * self.env.nrow
        # 用来存价值迭代之后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        """价值迭代"""
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 计算状态s下所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(self.v[s] - new_v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break  # 满足收敛，退出价值评估迭代
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):  # 根据价值函数导出贪婪策略
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa = p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            # 让最大Q值动作平分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


# 第五章时序差分


class CliffWalkingEnv_ch5:
    "悬崖漫步环境，用于Sarsa算法"

    def __init__(self, ncol, nrow):
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1

    def step(self, action):
        """更新机器人位置"""
        # 上下左右四个动作
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者终点
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态，坐标原点左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


## Sarsa算法


class Sarsa:
    """Sarsa算法"""

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon

    def take_action(self, state):
        """选取下一步操作，使用epsilon-贪婪算法"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """用来打印策略|最优动作"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]

        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


class nstep_Sarsa:
    """n步Sarsa算法"""

    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_action = n_action
        self.n = n  # n步Sarsa

        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    def take_action(self, state):
        """epsilon-贪婪算法采取动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """打印策略"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):
        """更新当前s0状态前面第n个状态的Q函数值
        先采样后面的n个数据，然后更新一开始状态的Q函数值
        """
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        if len(self.state_list) == self.n:  # 保存的数据足够n步更新
            G = self.Q_table[s1, a1]  # 得到 Q(s_{t+n},a_{t+n})

            for i in reversed(range(self.n)):
                G = G * self.gamma + self.reward_list[i]  # 不断向前计算每一步的回报
                # 如果到达终止状态，最后几步长度不够 n ，也将其更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            # 删除最前面的状态，下次更新下一个状态
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)

            # n 步 Sarsa 的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # 如果到达终止状态，即将开始下一条序列，则列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []


def print_agent_ch4(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print("⚠️⚠️⚠️⚠️", end=" ")
            elif (i * env.ncol + j) in end:
                print("✅✅✅✅", end=" ")
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ""
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else "🅾️"
                print(pi_str, end=" ")
        print()


## Q-learning 算法


class QLearning:
    """Q-learning 算法"""

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])
        self.epsilon = epsilon  # 贪婪策略参数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.n_action = n_action  # 动作个数

    def take_action(self, state):
        """epsilon-贪婪算法，选取下一步动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """用于打印策略|最优动作"""
        Qmax = max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Qmax:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        """更新"""
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


# 第六章 Dyna-Q算法


class CliffWalkingEnv_ch6:
    """冰湖环境，用于Dyna-Q算法"""

    def __init__(self, ncol, nrow):
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0  # 智能体初始位置
        self.y = self.nrow - 1

    def step(self, action):
        """更新智能体位置，通过传入的动作"""
        # 上下左右， 坐标原点左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))

        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.x > 0 and self.y == self.nrow - 1:
            done = True
            if self.x != self.ncol - 1:
                reward = -100

        return next_state, reward, done

    def reset(self):
        """重设环境，智能体位置复原"""
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


## Dynn-Q算法


class DynaQ:
    """Dyna-Q 算法"""

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])  # 初始化Q(s,a)表格
        self.epsilon = epsilon  # epsilon贪婪策略参数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.n_action = n_action  # 动作个数

        self.n_planning = (
            n_planning  # Q_planning 的次数，Q_planning是一步的，不是连续的
        )
        self.model = dict()  # 环境模型

    def take_action(self, state):
        """选取下一步动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        """Q-learning 更新"""
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        """更新 1 次 Q-learning 和 n 次 Q-planning"""
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # 将数据添加到环境模型
        # Q-planing 循环
        for _ in range(self.n_planning):
            # 随机尝试曾经遇到的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


def DynaQ_CliffWalking(n_planning):
    """Dyna-Q 训练函数"""
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv_ch6(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300  # 智能体在环境中运行的序列

    return_list = []  # 记录每一条序列的回报
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )

                pbar.update(1)
    return return_list


# 第七章 DQN算法
class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """将数据添加到buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """从buffer 中采样数据，数据量为 batch_size"""
        # 形状：[batch_size, 4]
        transitions = random.sample(self.buffer, batch_size)
        # 把解包之后的数据，按照状态，动作等串起来，成为元组
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """目前buffer中数据量"""
        return len(self.buffer)

    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done


class Qnet(nn.Module):
    """只有一层隐藏层的 Q 网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU 激活函数
        return self.fc2(x)


## DQN 算法


class DQN:
    """DQN 算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device,
    ):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 使用 Adam 优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率

        self.count = 0  # 计数，器记录更新次数
        self.device = device

    def take_action(self, state):
        """epsilon-贪婪策略选取动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        """更新"""
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        done = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        # 把动作对应的输出值 抠出来
        q_values = self.q_net(states).gather(1, actions)  # Q 值
        # 下个状态的最大 Q 值，这个只是数值，不参与 反向传播，
        # 里面的动作都是最大化取值，和actions没有关系
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # TD 误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - done)
        # 均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()  # 清零梯度
        dqn_loss.backward()  # 反向传播
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # 更新目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def get_gpu():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


class DQN_D:
    """DQN 算法， 包括Double DQN 算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device,
        dqn_type="VanillaDQN",
    ):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0
        self.dqn_type = dqn_type

    def take_action(self, state):
        """采取动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_values(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net(states).gather(1, actions)
        # DQN 和 Double DQN 区别
        if self.dqn_type == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_targets, q_values))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def dis_to_con(discrete_action, env, action_dim):
    """离散动作转回连续动作"""
    action_lowbound = env.action_space.low[0]  # 连续动作最小值
    action_upbound = env.action_space.high[0]  # 连续动作最大值
    return action_lowbound + (discrete_action / (action_dim - 1)) * (
        action_upbound - action_lowbound
    )


def train_dqn(agent, env, num_episode, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0

    for i in range(10):
        with tqdm(total=int(num_episode / 10), desc="Iterstion: %d" % i) as pbar:
            for episode_i in range(int(num_episode / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # 平滑处理
                    max_q_value = (
                        agent.max_q_values(state) * 0.005 + max_q_value * 0.995
                    )
                    # 保存每个状态下的最大Q值
                    max_q_value_list.append(max_q_value)
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, terminated, truncated, _ = env.step(
                        [action_continuous]
                    )
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "rewards": b_r,
                            "next_states": b_ns,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (episode_i + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episode / 10 * i + episode_i + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list, max_q_value_list


class VAnet(nn.Module):
    """只有一层隐藏层的A网络和V网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = (
            V + A - A.mean(1).view(-1, 1)
        )  # 使用V值和A值计算得到，减去均值保证模型稳定性
        return Q


class DQN_DD:
    """DQN算法，包罗Double DQN 和 Dueling DQN"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
        dqn_type="VanillaDQN",
    ):
        self.action_dim = action_dim
        if dqn_type == "DuelingDQN":
            # DUelingDQN网络架构不一样
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, action_dim)
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_values(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_targets, q_values))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# 第九章 REINFORCE


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    """ "REINFORCE算法"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.gamma = gamma
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        # 构建一个采样器，输入一个分布，方便从里面采样
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict["rewards"]
        state_list = transition_dict["states"]
        action_list = transition_dict["actions"]

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步开始计算
            reward = reward_list[i]
            state = (
                torch.tensor(state_list[i], dtype=torch.float)
                .unsqueeze(0)
                .to(self.device)
            )
            action = (
                torch.tensor(action_list[i]).unsqueeze(0).view(-1, 1).to(self.device)
            )
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降，上述梯度会累积


class ValueNet(nn.Module):
    """价值函数网络，输入state，输出value（一个数值）"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    """Actor-Critic 算法,这个实际是A2C，使用了优势函数"""

    def __init__(
        self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device
    ):
        # 策略网络 和 价值网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic = ValueNet(state_dim, hidden_dim)
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 优势函数 的 近似值，使用只是用V函数，来表示优势函数 A
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方损失误差
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


# 第十一章TRPO算法


def compute_advantage(gamma, lmbda, td_delta):
    """计算广义优势估计（GAE）函数，传入的td_delta是一整条轨迹的TD误差"""
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class TRPO:
    """TRPO 算法"""

    def __init__(
        self,
        hidden_dim,
        state_space,
        action_space,
        lmbda,
        kl_constraint,
        alpha,
        critic_lr,
        gamma,
        device,
    ):
        state_dim = state_space.shape[0]
        action_dim = action_space.n

        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE 参数
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

    def take_action(self, state):
        """离散型输出，选取动作，输出动作概率分布，然后从中采样一个动作"""
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """计算黑赛矩阵 和 一个向量的乘积"""
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 计算平均 KL 距离
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
        )
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL 距离的梯度先河向量进行进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """共轭梯度法求解方程"""
        # x 是初始方向,对应 v
        x = torch.zeros_like(grad)
        # 是残差
        r = grad.clone()
        # 是前一次的方向
        p = grad.clone()
        rdotr = torch.dot(r, r)

        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            # alpha 是步长
            alpha = rdotr / torch.dot(p, Hp)
            # 更新 估计
            x += alpha * p
            # 更新残差
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            # 计算beta ，保证共轭性更新方向
            beta = new_rdotr / rdotr
            # 更新搜索方向
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """计算策略目标"""
        log_probs = torch.log(actor(states).gather(1, actions))
        # 重要性采样的系数，模型输出值还需要进行softmax，分母都是1，重要性采样
        # 系数就是 log 输出相减
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
        self, states, actions, advantage, old_log_probs, old_action_dists, max_vec
    ):
        """线性搜索，更新网络参数，需要满足 散度小于限制值，且评估效果更好
        共轭梯度法得到优化方向，线性搜索得到优化步长 alpha ** i，搜索i看有没有合适的优化值
        """
        old_para = nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        # 先搜索一个步长，让模型参数学习，在比较两个模型输出的策略目标评估，需要满足策略评估更大
        for i in range(15):
            # 这个是步长，沿着共轭梯度求得的方向的步长
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )
            new_actor_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(
                torch.distributions.kl_divergence(old_action_dists, new_actor_dists)
            )
            # 新的策略目标，策略目标应该越大，说明学习到的东西更多
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, new_actor
            )
            # 策略目标更大，且满足KL散度限制
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        """更新策略函数"""
        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 共轭梯度法计算 x = H ^(-1) g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(
            states, old_action_dists, descent_direction
        )

        # 最大步长限制
        max_coef = torch.sqrt(
            2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8)
        )
        # 线性搜索
        new_para = self.line_search(
            states,
            actions,
            advantage,
            old_log_probs,
            old_action_dists,
            descent_direction * max_coef,
        )
        # 用线性搜索后的参数更新
        nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters()
        )

    def update(self, transition_dict):
        """更新"""
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # td 误差
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # softplus 是 平滑版的ReLU 保证输出的值是正数，标准差是正的
        std = F.softplus(self.fc_std(x))
        # tanh 输出值范围 [-1, 1] ， 乘 2 之后变成 [-2, 2]
        # 正好是动作的范围，保证动作的均值在[-2, 2]之间
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        # 高斯分布的均值和标准差
        return mu, std


class TRPOContinuous:
    """处理连续动作的TRPO算法"""

    def __init__(
        self,
        hiddem_dim,
        state_space,
        action_space,
        lmbda,
        kl_contraint,
        alpha,
        critic_lr,
        gamma,
        device,
    ):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        self.actor = PolicyNetContinuous(state_dim, hiddem_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hiddem_dim).to(device)
        self.cirtic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.alpha = alpha
        self.device = device
        self.kl_constraint = kl_contraint

    def take_action(self, state):
        """输入状态， 采取动作"""
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mu, std = self.actor(state)
        # 在 Pendulum-v0 中，动作是一个从[-2, 2]的范围取值，策略网络输出的是一个mu 和 std
        action_dist = torch.distributions.Normal(mu, std)
        return [action_dist.sample().item()]

    def hessian_matrix_vector_product(
        self, states, old_action_dists, vector, damping=0.1
    ):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
        )
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return (
            grad2_vector + damping * vector
        )  # 加上阻尼项，让梯度更加稳定，特别对于连续动作空间

    def conjugate_gradient(self, grad, states, old_action_dists):
        """共轭梯度，计算网络参数更新方向"""
        x = torch.zeros_like(grad)  # 最终的方向
        r = grad.clone()  # 残差，初始是 梯度
        p = grad.clone()  # x的更新方向

        rdotr = torch.dot(r, r)
        for i in range(10):  # 更新x的方向
            # Hp 是 曲率在 p方向上的投影
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            # x向 p 方向更新的步长，  (r*r)/(p*H*p)
            alpha = rdotr / torch.dot(p, Hp)

            x += alpha * p
            r -= alpha * Hp
            new_rdot_r = torch.dot(r, r)
            if new_rdot_r < 1e-10:  # 变化过小，结束查找x方向
                break
            # 缩放系数，保证当前更新方向p和之前的方向是垂直的|共轭的
            # 当前更新不会导致学习后退
            beta = new_rdot_r / rdotr
            p = r + beta * p
            rdotr = new_rdot_r
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """计算策略目标"""
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        #
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
        self, states, actions, advantage, old_log_probs, old_action_dists, max_vac
    ):
        """线性搜索，搜索网络参数的更新方向"""
        old_para = nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        for i in range(15):
            # 缩放系数，在最大更新方向上缩放
            coef = self.alpha**i
            # 新的网络参数
            new_para = old_para + coef * max_vac
            new_actor = copy.deepcopy(self.actor)
            nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )

            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            # kl散度
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
            )
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, new_actor
            )
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
            return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        desent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(
            states, old_action_dists, desent_direction
        )
        max_coef = torch.sqrt(
            2 * self.kl_constraint / torch.dot(desent_direction, Hd) + 1e-8
        )
        new_para = self.line_search(
            states,
            actions,
            advantage,
            old_log_probs,
            old_action_dists,
            max_coef * desent_direction,
        )
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters()
        )

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        rewards = (rewards + 8.0) / 8.0  # 修改奖励方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 对数概率密度值
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.cirtic_optimizer.zero_grad()
        critic_loss.backward()
        self.cirtic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)


# PPO 算法


class PPO:
    """PPO算法， 采用截断方式"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        # GAE 广义优势函数
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        """
        这里的 循环 是和TRPO最大的不同
        TRPO 使用 共轭梯度和线性搜索，来查找新的参数更新方向
        一是因为黑赛矩阵计算量很大；二是不知道之后策略的数据分布

        PPO 这里省略掉了探索后续策略参数的过程，改成迭代拟合的方式
        初始分布就是原本数据，后续逐渐迭代分布会发生变化
        支持它能够迭代的是他的损失函数，采用 截断ratio 的advantage
        因为 advantag也是估计的，数值不准，这样截断牺牲一点 bias,减少方差van
        """ ""
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            # 截断
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # PPO 损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()


class PPOCountinuous:
    """处理连续动作的PPO算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    ):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).squeeze(0).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        rewards = (rewards + 8) / 8.0
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 这个是采样数据|前面策略的数据分布的 对数化概率密度
        # 连续空间中无法求出概率，使用概率密度计算后的重要性采样，
        # 再做exp转换就可以作为ratio
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# 第 13 章 DDPG 算法
"""
PPO 和 TRPO 是随机策略，因为策略网络输出的 mu 和 std,输出的是一个分布,还需要从中采样
DDPG 是确定性策略，策略网络直接输出 动作的数值
"""


class PolicyNet_ch13(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 环境可以接受的最大值 这里[-2, 2]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)


class TwoLayerFC(nn.Module):
    def __init__(
        self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x: x
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)

        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    """DDPG 算法"""

    def __init__(
        self,
        num_in_actor,
        num_out_actor,
        num_in_critic,
        hidden_dim,
        discrete,
        action_bound,
        sigma,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        device,
    ):
        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(
            num_in_actor, num_out_actor, hidden_dim, F.relu, out_fn
        ).to(device)
        self.target_actor = TwoLayerFC(
            num_in_actor, num_out_actor, hidden_dim, F.relu, out_fn
        ).to(device)
        self.critic = TwoLayerFC(num_in_critic, 1, hidden_dim, F.relu).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 1, hidden_dim, F.relu).to(device)
        # 初始化目标价值网络，设置相同参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声标准差，均值设为 0
        self.tau = tau  # 目标网络软更新参数
        self.action_bound = action_bound  # 环境接受的动作最大值
        self.device = device
        self.action_dim = num_out_actor

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).item()
        # 添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_updata(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        next_q_values = self.target_critic(
            torch.cat([next_states, self.target_actor(next_states)], dim=1)
        )

        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(
            F.mse_loss(self.critic(torch.cat([states, actions], dim=1)), q_targets)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor 只需要最大化Q
        actor_loss = -torch.mean(
            self.critic(torch.cat([states, self.actor(states)], dim=1))
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新 目标网络
        self.soft_updata(self.actor, self.target_actor)
        self.soft_updata(self.critic, self.target_critic)


# 第十六章 模型预测控制 MPC


class CEM:
    """交叉熵选取动作"""

    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound, lower_bound):
        self.n_sequence = n_sequence
        self.elite_ratio = elite_ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fake_env = fake_env  # 模拟环境的模型

    def optimizer(self, state, init_mean, init_var):
        """选取动作，通过优化五次，选出一个动作序列"""
        mean = init_mean
        var = init_var
        # X 是在生成了一个 action_dim * 动作长度 长度的一个 分布
        # 这个分布使用了截断 保证 采样在 mean +- 2*sigmoid 范围中
        # 分布的 mean 和var 使用np矩阵表示
        # 后面直接采样 就可以的到这一连串动作的每一个动作维度的 动作数据
        # 然后评估动作来优化 这个序列的 mean，来优化下一次动作序列的选择
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        # 初始状态， 复制 序列数 个个数，每一次优化使用 n_sequence 条动作序列
        state = np.tile(state, (self.n_sequence, 1))

        for _ in range(5):
            # 得到 当前均值 到上界和下界的距离
            # 采样时 需要限制 方差，保证当mean 靠近上下界的时候 采样的的大部分数据不会越界
            # 因此需要让 均值到 边界的距离 保持在 2 * sigmoid ，让 95% 的采样数据在边界内
            # 反向推导出 方差 是 contrained_var
            # 只需要更靠近的哪一个边界距离 满足就可以，然后求出方差的限制，
            # 真实的采样方差不能超过这个限制，得到方差用于后续采样
            lb_list, ub_list = mean - self.lower_bound, self.upper_bound - mean

            contrained_var = np.minimum(
                np.minimum(np.square(lb_list / 2), np.square(ub_list / 2)), var
            )

            # 生成动作序列
            action_sequences = (
                X.rvs(size=(self.n_sequence, mean.shape[0])) * np.sqrt(contrained_var)
                + mean
            )
            # action_sequences = [X.rvs() for _ in range(self.n_sequence)] * np.sqrt(
            #   contrained_var
            # ) + mean
            # 计算动作序列的累计奖励, 对n条动作序列，进行模拟，得到模型评估奖励
            returns = self.fake_env.propagate(state, action_sequences)[:, 0]
            # 选取奖励最高的若干条动作序列 作为精英|elite序列
            elites = action_sequences[np.argsort(returns)][
                -int(self.elite_ratio * self.n_sequence) :
            ]
            # elite 序列每一个维度的均值
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            # 更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var
        return mean  # 返回的mean，直接代表选取的动作的动作值，其中包括每个动作维度，所有动作的动作维度平摊成 1维


class Swish(nn.Module):
    """Swish 激活函数"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def init_weights(m):
    """初始化模型权重"""

    def truncated_normal_init(t, mean=0.0, std=0.01):
        """截断式初始化"""
        with torch.no_grad():
            tmp = torch.randn_like(t) * std + mean
            while True:
                cond = (tmp < mean - 2 * std) | (tmp > mean + 2 * std)
                if not cond.any():
                    break
                tmp = torch.where(
                    cond,
                    torch.randn_like(tmp) * std + mean,  # 重新采样的值
                    tmp,  # 保留不需要替换的部分
                )
            return tmp.to(t.device)

    if isinstance(m, nn.Linear) or isinstance(m, FCLayer):
        std = 1 / (2 * np.sqrt(m._input_dim))
        m.weight.data.copy_(truncated_normal_init(m.weight, std=std))
        m.bias.data.fill_(0.0)


class FCLayer(nn.Module):
    """集成之后的全连接层"""

    def __init__(self, input_dim, output_dim, ensemble_size, activation, device):
        super().__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, input_dim, output_dim).to(device)
        )
        self._activation = activation
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_dim).to(device))

    def forward(self, x):
        # 输入  集成模型数量， 输入维度
        # 参数w维度   集成模型数量， 输入维度， 输出维度
        # 使用 bmm 保持第一个维度
        return self._activation(
            torch.add(torch.bmm(x, self.weight), self.bias[:, None, :])
        )


class EnsembleModel(nn.Module):
    """环境集成, 环境模型"""

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        ensemble_size=5,
        learning_rate=1e-3,
    ):
        super().__init__()
        # 输出包括每一个动作维度的方差和均值，还有奖励的
        self._output_dim = (state_dim + 1) * 2
        # 方差对数后的限制 ，初始最大是 0.5， 最小 -10
        self._max_logvar = nn.Parameter(
            (torch.ones((1, self._output_dim // 2)).float() / 2).to(device),
            requires_grad=False,
        )
        self._min_logvar = nn.Parameter(
            (-torch.ones((1, self._output_dim // 2)).float() * 10).to(device),
            requires_grad=False,
        )

        self.layer1 = FCLayer(
            state_dim + action_dim, 200, ensemble_size, Swish(), device
        )
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish(), device)
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish(), device)
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish(), device)
        self.layer5 = FCLayer(
            200, self._output_dim, ensemble_size, nn.Identity(), device
        )
        self.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        mean = ret[:, :, : self._output_dim // 2]
        # PETS算法中， 将方差控制在最大最小值之间
        logvar = self._max_logvar - F.softplus(
            self._max_logvar - ret[:, :, self._output_dim // 2 :]
        )
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)

        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logver, labels, use_var_loss=True):
        # 方差相反数，为负数
        inverse_var = torch.exp(-logver)
        if use_var_loss:
            # 高斯分布下的 负对数似然， mse_loss + 1/2 * var_loss, 这里省略系数 1/2
            # 维度 Batch_size, 时间步|horizon, action_dim ==> Batch_size
            mse_loss = torch.mean(
                torch.mean(torch.pow(mean - labels, 2) * inverse_var, dim=-1), dim=-1
            )
            var_losss = torch.mean(torch.mean(logver, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_losss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(-1, -2))
            total_loss = torch.sum(mse_loss)

        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        # loss 加上 logvar 的上下界 的L1范数作为正则项, 约束 上下界不要过大
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    """环境模型集成， 加入精细化的训练"""

    def __init__(self, state_dim, action_dim, device, num_network=5):
        self._num_network = num_network
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.model = EnsembleModel(
            state_dim, action_dim, device, ensemble_size=num_network
        )
        self._epoch_since_last_update = 0
        self.device = device

    def train(self, inputs, labels, batch_size=64, holdout_ratio=0.1, max_iter=20):
        # 设置训练集 和 验证集
        permutation = np.random.permutation(inputs.shape[0])  # 生成随机索引
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)  # 验证集数量
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])

        # 保存每个模型中的最好结果，保存为元组 (迭代次数， loss)
        self._snapshots = {i: (None, 1e10) for i in range(self._num_network)}

        for epoch in itertools.count():  # 无穷循环
            train_index = np.stack(
                [
                    np.random.permutation(train_inputs.shape[0])
                    for _ in range(self._num_network)
                ]
            )
            # 所有真实数据都用来训练
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                batch_index = train_index[
                    :, batch_start_pos : batch_start_pos + batch_size
                ]
                train_input = (
                    torch.from_numpy(train_inputs[batch_index]).float().to(self.device)
                )
                train_label = (
                    torch.from_numpy(train_labels[batch_index]).float().to(self.device)
                )

                mean, logvar = self.model(train_input, return_log_var=True)
                loss, _ = self.model.loss(mean, logvar, train_label)
                self.model.train(loss)

            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_log_var=True)
                _, holdout_losses = self.model.loss(
                    mean, logvar, holdout_labels, use_var_loss=False
                )
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(epoch, holdout_losses)
                # 结束条件 1. 五次没有更新 _snapshots ,学习之后没有更好的表现
                #         2. 达到最大迭代次数 20
                if break_condition or epoch > max_iter:
                    break

    def _save_best(self, epoch, losses, threshold=0.1):
        """训练之后，使用验证集，如果提升大于阈值，更新每个模型的最优结果"""
        updated = False
        for i in range(len(losses)):
            current = losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > threshold:
                self._snapshots[i] = (epoch, current)  # 更新最优结果
                updated = True
            self._epoch_since_last_update = (
                0 if updated else self._epoch_since_last_update + 1
            )
            return self._epoch_since_last_update > 5  # 五次没有更新 结束当前训练

    def predict(self, inputs, batch_size=64):
        """在进行 评估 的时候,预测采样的 reward 和动作的 mean 和 var"""
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i : i + batch_size]).float().to(self.device)
            cur_mean, cur_var = self.model(
                input[None, :, :].repeat([self._num_network, 1, 1]),
                return_log_var=False,
            )
            mean.append(cur_mean.detach().cpu().numpy())
            var.append(cur_var.detach().cpu().numpy())
        return np.hstack(mean), np.hstack(var)


class FakeEnv:
    """使用模型进行动作评估"""

    def __init__(self, model):
        self.model = model

    def step(self, obs, act):
        """这是一个序列的输入，"""
        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[
            :, :, 1:
        ] += obs.numpy()  # 输出的mean是偏移量，需要加上原本的值
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        # 根据 均值和方差构建的正态分布 进行采样
        ensemble_samples = (
            ensemble_model_means
            + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
        )

        num_models, batch_size, _ = ensemble_model_means.shape
        # 从 所有模型中，为每一个样本 随机 选择一个模型
        models_to_use = np.random.choice(
            [i for i in range(self.model._num_network)], size=batch_size
        )

        batch_inds = np.arange(0, batch_size)
        # ensemble_samples 形状 模型数量, batch_size, 输出维度
        samples = ensemble_samples[models_to_use, batch_inds]
        # samples 形状  batch_size, 输出维度
        # 其中输出维度 是  1（奖励） + 状态维度
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        return rewards, next_obs

    def propagate(self, obs, actions):
        with torch.no_grad():
            obs = np.copy(obs)
            total_reward = np.expand_dims(np.zeros(obs.shape[0]), axis=-1)
            obs, actions = torch.as_tensor(obs), torch.as_tensor(actions)
            for i in range(actions.shape[1]):
                action = torch.unsqueeze(actions[:, i], 1)
                rewards, next_obs = self.step(obs, action)
                total_reward += rewards
                obs = torch.as_tensor(next_obs)
            return total_reward


class PETS:
    """PETS 算法"""

    def __init__(
        self,
        env: gym.Env,
        buffer_size,
        n_sequence,
        elite_raito,
        plan_horizon,
        num_episodes,
        device,
    ):
        self._env = env
        self._env_pool = ReplayBuffer(buffer_size)

        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim, device)
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        self._cem = CEM(
            n_sequence, elite_raito, self._fake_env, self.upper_bound, self.lower_bound
        )
        self.plan_horizon = plan_horizon
        self.num_episodes = num_episodes

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = env_samples[0]
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = env_samples[3]
        inputs = np.concatenate([obs, actions], axis=-1)
        labels = np.concatenate([rewards, next_obs - obs], axis=-1)
        self._model.train(inputs, labels)

    def mpc(self):
        """模型预测控制, 用来进行训练一次环境模型之后的 评估"""
        # 初始化 mean 和 var
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0, self.plan_horizon)
        var = np.tile(
            np.square(self.upper_bound - self.lower_bound) / 16, self.plan_horizon
        )
        obs, _ = self._env.reset(seed=0)
        done, episode_return = False, 0
        while not done:
            # 动作是根据 fake_env 的评估从 分布中采样处来的
            # 动作个数 和 维度 已经隐含在 mean 和 var 中
            actions = self._cem.optimizer(obs, mean, var)
            action = actions[: self._action_dim]  # 选取第一个动作
            next_obs, reward, terminated, truncated, _ = self._env.step(
                action
            )  # 真实环境
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            # 去除第一个动作，在后面添加空白
            mean = np.concatenate(
                [
                    np.copy(actions)[self._action_dim :],
                    np.zeros(self._action_dim),
                ]
            )
        return episode_return

    def explore(self):
        """探索一次， 生成一些数据"""
        obs, _ = self._env.reset()
        done, episode_return = False, 0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self._env.step(
                action
            )  # 真实环境
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
        return episode_return

    def train(self):
        return_list = []
        explore_return = self.explore()  # 进行随机策略探索来首集一条序列的数据
        print("episode: 1, return: %d" % explore_return)
        return_list.append(explore_return)

        for i_episode in range(self.num_episodes - 1):
            self.train_model()
            episode_return = self.mpc()
            return_list.append(episode_return)
            print("episode: %d, return: %d" % (i_episode + 2, episode_return))
        return return_list


# 第十四章 SAC算法


class PolicyNetContinuous_ch14(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # 约束 std 非负
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()  # rsample() 是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        # 压缩 动作取值在[-1, 1]，在乘上action_bound 即可
        # 压缩后 log 概率密度也要变换，使用 概率密度修正公式
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度,使用概率密度修正公式
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound  # 得到真正的动作
        return action, log_prob


class QValueNetContinuous_ch14(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    """处理连续动作的SAC算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        self.actor = PolicyNetContinuous_ch14(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        # 两个Q网络
        self.critic_1 = QValueNetContinuous_ch14(state_dim, hidden_dim, action_dim).to(
            device
        )
        self.critic_2 = QValueNetContinuous_ch14(state_dim, hidden_dim, action_dim).to(
            device
        )
        # 两个目标Q网络
        self.target_critic_1 = QValueNetContinuous_ch14(
            state_dim, hidden_dim, action_dim
        ).to(device)
        self.target_critic_2 = QValueNetContinuous_ch14(
            state_dim, hidden_dim, action_dim
        ).to(device)

        # 目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        # 使用 alpha 的 log值，可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        """计算目标Q值"""
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        # 两个目标网络 取 更小的
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)

        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_updata(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # 重塑奖励，方便训练 [-16, 0] -> [-1, 1]
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach())
        )

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(
            -self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value)
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 软更新 目标网络
        self.soft_updata(self.critic_1, self.target_critic_1)
        self.soft_updata(self.critic_2, self.target_critic_2)


class PolicyNet_ch14(nn.Module):
    """离散动作策略网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QvalueNet_ch14(nn.Module):
    """离散动作的 Q 价值函数"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC:
    """处理离散动作的SAC算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        # 策略网络
        self.actor = PolicyNet_ch14(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QvalueNet_ch14(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QvalueNet_ch14(state_dim, hidden_dim, action_dim).to(device)

        # 目标网络
        self.target_critic_1 = QvalueNet_ch14(state_dim, hidden_dim, action_dim).to(
            device
        )
        self.target_critic_2 = QvalueNet_ch14(state_dim, hidden_dim, action_dim).to(
            device
        )

        # 令 Q目标网络 初始参数和 Q网络 一致
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        # 使用alpha 的log值，使训练更稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True

        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(
            next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True
        )
        # 目标值加入 交叉熵约束
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        # 动作不是 float类型
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_2_q_values = self.critic_2(states).gather(1, actions)

        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        # 后面的 1e-8 很重要 防止 actor输出 NAN
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(log_probs * probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        # 直接根据概率计算期望
        min_qvalue = torch.sum(
            probs * torch.min(q1_value, q2_value), dim=1, keepdim=True
        )
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
