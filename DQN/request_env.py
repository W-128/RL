import datetime

import numpy as np
import pandas as pd
import time
import csv
from common.utils import make_dir
import os

# t=100ms
TIME_UNIT = 0.1
TIME_UNIT_IN_ON_SECOND = int(1 / TIME_UNIT)
# 引擎能承受的单位时间最大并发量
THRESHOLD = int(40 / TIME_UNIT_IN_ON_SECOND)

# 状态向量的维数/rtl的级别个数+1
STATE_DIMENSION = 6
# state=(阈值,剩余时间为1的请求个数,...,剩余时间为5的请求个数)
# 动作空间维数 == 状态向量的维数
ACTION_SPACE_DIMENSION = STATE_DIMENSION

# 实时用的话，这个地方无法事先写好，只能每秒来append
# 现在先 直接从文件读取

# request=[request_id, arrive_time, rtl, remaining_time]
# end_request[request_id, arrive_time, rtl, wait_time]
REQUEST_ID_INDEX = 0
ARRIVE_TIME_INDEX = 1
RTL_INDEX = 2
REMAINING_TIME_INDEX = 3
WAIT_TIME_INDEX = 3
NEW_ARRIVE_REQUEST_IN_DIC = []

FRESH_TIME = 1
NEED_EVALUATE_ENV_CORRECT = True

# t t的长度为TIME_UNIT
t = 0
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


def t_add_one():
    global t
    t = t + 1


def t_to_zero():
    global t
    t = 0


# 模拟环境的返回次数，最多返回阈值次
SIM_ENV_USE_TIME = 0


def SIM_ENV_USE_TIME_add_one():
    global SIM_ENV_USE_TIME
    SIM_ENV_USE_TIME = SIM_ENV_USE_TIME + 1


def SIM_ENV_USE_TIME_to_zero():
    global SIM_ENV_USE_TIME
    SIM_ENV_USE_TIME = 0


'''
将数据集转换为arriveTime_request_dic
arriveTime_request_dic:
key=arriveTime
value=arriveTime为key的request_in_dic列表
request_in_dic的形式为[request_id, arrive_time, rtl]
'''
filename = 'concurrent_request_num.csv'
data = pd.read_csv(filename, header=0)
for i in range(0, len(data)):
    request_in_dic = [data.loc[i, 'request_id'], data.loc[i, 'arrive_time'], data.loc[i, 'rtl']]
    NEW_ARRIVE_REQUEST_IN_DIC.append(request_in_dic)

arriveTime_request_dic = {}
for request_in_dic in NEW_ARRIVE_REQUEST_IN_DIC:
    if request_in_dic[ARRIVE_TIME_INDEX] in arriveTime_request_dic:
        arriveTime_request_dic[request_in_dic[ARRIVE_TIME_INDEX]].append(request_in_dic)
    else:
        request_list = [request_in_dic]
        arriveTime_request_dic[request_in_dic[ARRIVE_TIME_INDEX]] = request_list


class RequestEnv:
    def __init__(self):
        self.action_space = []
        for i in range(STATE_DIMENSION):
            self.action_space.append(str(i))
        self.n_actions = len(self.action_space)
        self.action_space_dimension = ACTION_SPACE_DIMENSION
        self.state_dimension = STATE_DIMENSION
        '''
        [剩余时间为[0,1s)的请求列表,剩余时间为[1s,2s)...,剩余时间为[4s,5s)的请求列表]
        active_request_group_by_remaining_time_list是中间变量，随时间推移会有remainingTime的改变
        '''
        self.active_request_group_by_remaining_time_list = []
        for i in range(STATE_DIMENSION - 1):
            self.active_request_group_by_remaining_time_list.append([])
        self.state_record = []
        self.end_request_list = []
        self.fail_request_list = []
        self.simulate_time = 0
        self.episode = 0
        self.end_request_result_path = curr_path + '/end_request/' + curr_time + '/'
        make_dir(self.end_request_result_path)

    def get_not_avail_actions(self):
        not_avail_actions = []
        for i in range(1, len(self.state_record)):
            if self.state_record[i] == 0:
                not_avail_actions.append(i)
        return not_avail_actions

    # 返回奖励值和下一个状态
    def step(self, action):
        '''
        debug
        print('action: ' + str(action))
        print('t:' + str(t))
        '''
        reward = self.get_reward(action)
        # 环境更新
        # sim_env更新
        done = self.update_sim_env(action)
        SIM_ENV_USE_TIME_add_one()
        # 最后一个模拟状态的更新值要和新到来的合并成S_{t+1}发送给agent
        if SIM_ENV_USE_TIME == THRESHOLD:
            SIM_ENV_USE_TIME_to_zero()
        done = self.update_env()
        # 验证环境正确性
        if done and NEED_EVALUATE_ENV_CORRECT:
            print('环境正确性:' + str(self.is_correct()))
        '''
        # debug
        print('active_request_list:' + str(self.active_request_group_by_remaining_time_list))
        '''
        return self.state_record, reward, done

    def is_correct(self):
        all_request_id_list = []
        for request_in_dic in NEW_ARRIVE_REQUEST_IN_DIC:
            all_request_id_list.append(request_in_dic[REQUEST_ID_INDEX])
        all_request_after_episode_list = []
        all_request_id_after_episode_list = []
        for request in self.fail_request_list:
            all_request_after_episode_list.append(request)
            all_request_id_after_episode_list.append(request[REQUEST_ID_INDEX])
        for request in self.end_request_list:
            all_request_after_episode_list.append(request)
            all_request_id_after_episode_list.append(request[REQUEST_ID_INDEX])
        all_request_id_list.sort()
        all_request_id_after_episode_list.sort()
        return all_request_id_after_episode_list == all_request_id_list

    def get_success_rate(self):
        all_request = []
        for request_in_dic in NEW_ARRIVE_REQUEST_IN_DIC:
            all_request.append(request_in_dic[REQUEST_ID_INDEX])
        all_request_num = all_request.__len__()
        return self.end_request_list.__len__() / all_request_num

    def save_success_request(self):
        # end_request[request_id, arrive_time, rtl, wait_time]
        headers = ['request_id', 'arrive_time', 'rtl', 'wait_time']
        with open(self.end_request_result_path + 'end_request' + str(self.episode) + '.csv', 'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(self.end_request_list)

    # request_list -> state
    def active_request_group_by_remaining_time_list_to_state(self):
        state = [THRESHOLD]
        for active_request_group_by_remaining_time in self.active_request_group_by_remaining_time_list:
            state.append(len(active_request_group_by_remaining_time))

        self.state_record = state

    def update_env(self):
        # active_request_group_by_remaining_time_list 剩余时间要推移
        for i in range(len(self.active_request_group_by_remaining_time_list)):
            for j in range(len(self.active_request_group_by_remaining_time_list[i])):
                self.active_request_group_by_remaining_time_list[i][j][REMAINING_TIME_INDEX] = \
                    self.active_request_group_by_remaining_time_list[i][j][REMAINING_TIME_INDEX] - 1

        for i in range(len(self.active_request_group_by_remaining_time_list)):
            for active_request in self.active_request_group_by_remaining_time_list[i][:]:
                # 过期请求
                if active_request[REMAINING_TIME_INDEX] == 0:
                    self.fail_request_list.append(list(active_request))
                    self.active_request_group_by_remaining_time_list[i].remove(active_request)
                if active_request[REMAINING_TIME_INDEX] // TIME_UNIT_IN_ON_SECOND != i:
                    self.active_request_group_by_remaining_time_list[i - 1].append(list(active_request))
                    self.active_request_group_by_remaining_time_list[i].remove(active_request)

        episode_done = False
        # 与真实环境交互的话这里需要更改
        new_arrive_request_list = self.get_new_arrive_request_list()
        # 更新active_request 为active_request_{t+1}
        for i in range(len(self.active_request_group_by_remaining_time_list)):
            self.active_request_group_by_remaining_time_list[i] = self.active_request_group_by_remaining_time_list[i] + \
                                                                  new_arrive_request_list[i]
        # 状态更新
        self.active_request_group_by_remaining_time_list_to_state()
        # 判断是否结束
        remaining_request_is_done = True
        for i in range(1, len(self.state_record)):
            if self.state_record[i] != 0:
                remaining_request_is_done = False
        if t > np.max(list(arriveTime_request_dic.keys())) and remaining_request_is_done:
            episode_done = True
        # time.sleep(FRESH_TIME)
        return episode_done

    def update_sim_env(self, action):
        if action != 0:
            # 确保 action 有mask 不会选择队列为空的剩余时间队列
            time_stamp = time.time()
            '''
            todo
            选择在剩余时间(A-1,A]中最小的
            '''
            submit_index = np.random.choice(self.active_request_group_by_remaining_time_list[action - 1].__len__())
            end_request = list(self.active_request_group_by_remaining_time_list[action - 1][submit_index])
            # 把提交的任务从active_request_list中删除
            del self.active_request_group_by_remaining_time_list[action - 1][submit_index]
            end_request[WAIT_TIME_INDEX] = t - end_request[ARRIVE_TIME_INDEX]
            self.end_request_list.append(end_request)
            self.state_record[action] = self.state_record[action] - 1
        return False

    def get_reward(self, action):
        if action == 1:
            reward = -(self.state_record[1] - 1)
        else:
            if action == 0:
                reward = -(self.state_record[1])
            else:
                reward = -(self.state_record[1] + action - 1)
        return reward

    # 现在用t来表示，真实环境中收集[t-1,t)到来的请求直接给出
    def get_new_arrive_request_list(self):
        now_new_arrive_request_list = []
        for i in range(STATE_DIMENSION - 1):
            now_new_arrive_request_list.append([])
        if t in arriveTime_request_dic:
            for request_in_dic in arriveTime_request_dic[t]:
                # request_in_dic的形式为[request_id, arrive_time, rtl]
                # request [request_id, arrive_time, rtl, remaining_time]
                # request_in_dic 转为request
                request = list(request_in_dic)
                # 刚加进缓冲时 remaining_time=rtl
                request.append(request_in_dic[RTL_INDEX])
                now_new_arrive_request_list[request[REMAINING_TIME_INDEX] // TIME_UNIT_IN_ON_SECOND].append(request)
        t_add_one()
        return now_new_arrive_request_list

    #   初始状态
    def reset(self, episode):
        self.episode = episode
        self.simulate_time = 0
        t_to_zero()
        self.end_request_list = []
        self.fail_request_list = []
        self.active_request_group_by_remaining_time_list = self.get_new_arrive_request_list()
        self.active_request_group_by_remaining_time_list_to_state()
        return self.state_record

    def get_active_request_sum(self):
        sum = 0
        for active_request in self.active_request_group_by_remaining_time_list:
            sum += active_request.__len__()
        return sum
