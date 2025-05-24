"""
r2l.py - 动手学强化学习的包，这是动手学强化学习的练习代码，包含其中使用的部分代码

Author: github@AustinSuun
Data: 2025-05-24
"""

# __version__= "0.1"

# 核心库
import copy

# 三方库
import numpy


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
