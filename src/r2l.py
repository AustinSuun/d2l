"""
r2l.py - 动手学强化学习的包，这是动手学强化学习的练习代码，包含其中使用的部分代码

Author: github@AustinSuun
Data: 2025-05-24
"""

# __version__= "0.1"

# 核心库
import copy

# 三方库
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm  # 显示循环进度条

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
