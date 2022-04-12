import numpy as np
import pandas as pd
import math
from collections import defaultdict
import torch
import math


class QLearningTable:
    def __init__(self, actions, cfg):
        self.actions = actions  # a list
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma
        self.epsilon = 0
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

    def set_q_table(self, key, value):
        self.q_table[key] = value

    def print_q_table(self):
        print('q_table:')
        for k in self.q_table.keys():
            print(k + ' ' + str(self.q_table[k]))

    def choose_action(self, observation, not_avail_action):
        '''
        # debug
        print('observation: ' + str(observation))
        print('not_avail_action: ' + str(not_avail_action))
        '''
        # 排除掉非法动作
        if len(not_avail_action) != 0:
            for action in not_avail_action:
                self.q_table[str(observation)][action] = -float("inf")
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table[str(observation)]
            best_q_value = np.max(state_action)
            choose_action_list = np.argwhere(state_action == best_q_value).flatten().tolist()
            '''
            # debug
            print('state_action: ' + str(state_action))
            print('best_q_value: ' + str(best_q_value))
            print("choose_action_list: " + str(choose_action_list))
            print("choose_action_list:" + choose_action_list)
            '''
            action = np.random.choice(choose_action_list)
        else:
            # choose random action
            avail_action = []
            for action in self.actions:
                if not_avail_action.count(action) == 0:
                    avail_action.append(action)
            action = np.random.choice(avail_action)
        # self.print_q_table()
        return action

    def predict(self, observation, not_avail_action):
        # 排除掉非法动作
        if len(not_avail_action) != 0:
            for action in not_avail_action:
                self.q_table[str(observation)][action] = -float("inf")
        action = np.random.choice(np.argwhere(
            self.q_table[str(observation)] == np.max(self.q_table[str(observation)])).flatten().tolist())
        return action

    def learn(self, s, a, r, s_, done):
        q_predict = self.q_table[str(s)][a]
        if done:
            q_target = r  # next state is terminal
        else:
            q_target = r + self.gamma * np.max(self.q_table[str(s_)])  # next state is not terminal
        self.q_table[str(s)][a] += self.lr * (q_target - q_predict)
        if math.isnan(self.q_table[str(s)][a]):
            self.q_table[str(s)][a] = -float("inf")
        # debug
        if math.isnan(self.q_table[str(s)][a]):
            print('s: ' + str(s))
            print('s_: ' + str(s_))
            print('self.q_table[str(s_)]' + str(self.q_table[str(s_)]))
            print('np.max(self.q_table[str(s_)]):' + str(np.max(self.q_table[str(s_)])))
            print('q_target: ' + str(q_target))
            print('q_predict:' + str(q_predict))

    def save(self, path):
        import dill
        torch.save(
            obj=self.q_table,
            f=path + "Qleaning_model.pkl",
            pickle_module=dill
        )
        print("保存模型成功！")

    def load(self, path):
        import dill
        self.q_table = torch.load(f=path + 'Qleaning_model.pkl', pickle_module=dill)
        print("加载模型成功！")
