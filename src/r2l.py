"""
r2l.py - 动手学强化学习的包，这是动手学强化学习的练习代码，包含其中使用的部分代码

Author: github@AustinSuun
Data: 2025-05-24
"""

__version__ = "0.1"

# 核心库
import copy
import random

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
