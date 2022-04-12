import datetime

import numpy as np
import pandas as pd
import time
import csv
from my_common.utils import make_dir
from my_common.get_data import get_arrive_time_request_dic
import os

# t=1000ms
TIME_UNIT = 1
TIME_UNIT_IN_ON_SECOND = int(1 / TIME_UNIT)
# 引擎能承受的单位时间最大并发量
THRESHOLD = int(40 / TIME_UNIT_IN_ON_SECOND)

# 状态向量的维数/rtl的级别个数+1
STATE_DIMENSION = 7
# state=(剩余时间为0的请求个数,...,剩余时间为5的请求个数,阈值)
# 动作空间维数 == 状态向量的维数
ACTION_SPACE_DIMENSION = STATE_DIMENSION

# 实时用的话，这个地方无法事先写好，只能每秒来append
# 现在先 直接从文件读取

# request=[request_id, arrive_time, rtl, remaining_time]
# success_request_list[request_id, arrive_time, rtl, wait_time]
REQUEST_ID_INDEX = 0
ARRIVE_TIME_INDEX = 1
RTL_INDEX = 2
REMAINING_TIME_INDEX = 3
WAIT_TIME_INDEX = 3

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


class RequestEnv:

    def __init__(self):
        # [0,1,...,5,null]
        self.action_space = []
        for i in range(STATE_DIMENSION - 1):
            self.action_space.append(str(i))
        self.action_space.append('null')
        self.action_space_dimension = len(self.action_space)
        '''
        [剩余时间为0s的请求列表,剩余时间为1s...,剩余时间为5s的请求列表]
        active_request_group_by_remaining_time_list是中间变量，随时间推移会有remainingTime的改变
        '''
        self.active_request_group_by_remaining_time_list = []
        for i in range(STATE_DIMENSION - 1):
            self.active_request_group_by_remaining_time_list.append([])
        self.state_record = []
        self.success_request_list = []
        self.fail_request_list = []
        self.simulate_time = 0
        self.episode = 0
        '''
        arriveTime_request_dic:
        key=arriveTime
        value=arriveTime为key的request_in_dic列表
        request_in_dic的形式为[request_id, arrive_time, rtl]
        '''
        self.new_arrive_request_in_dic, self.arriveTime_request_dic = get_arrive_time_request_dic(
            ARRIVE_TIME_INDEX)
        self.end_request_result_path = curr_path + '/success_request_list/' + curr_time + '/'
        make_dir(self.end_request_result_path)

    # 返回奖励值和下一个状态
    def step(self, action):

        # debug
        # print('action: ' + str(action))
        # print('t:' + str(t))

        # 环境更新
        # sim_env更新
        done = self.update_sim_env(action)
        SIM_ENV_USE_TIME_add_one()
        # 最后一个模拟状态的更新值要和新到来的合并成S_{t+1}发送给agent
        if SIM_ENV_USE_TIME == THRESHOLD:
            SIM_ENV_USE_TIME_to_zero()
            t_add_one()
            done = self.update_env()
        reward = self.get_reward(action)

        # 验证环境正确性
        if done and NEED_EVALUATE_ENV_CORRECT:
            print('环境正确性:' + str(self.is_correct()))

        # debug
        # print('active_request_list:' + str(self.active_request_group_by_remaining_time_list))

        return self.state_record, reward, done

    def update_env(self):
        # remaining_time==0且还留在active_request_group_by_remaining_time_list中的请求此时失败
        for active_request in self.active_request_group_by_remaining_time_list[
                0]:
            self.fail_request_list.append(list(active_request))
        self.active_request_group_by_remaining_time_list[0] = []

        # active_request_group_by_remaining_time_list 剩余时间要推移
        for i in range(
                1, self.active_request_group_by_remaining_time_list.__len__()):
            self.active_request_group_by_remaining_time_list[i - 1] = []
            for active_request in self.active_request_group_by_remaining_time_list[
                    i]:
                active_request[REMAINING_TIME_INDEX] = active_request[
                    REMAINING_TIME_INDEX] - 1
                self.active_request_group_by_remaining_time_list[i - 1].append(
                    list(active_request))

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
        for i in range(0, len(self.state_record) - 1):
            if self.state_record[i] != 0:
                remaining_request_is_done = False
        if t > np.max(list(self.arriveTime_request_dic.keys())
                      ) and remaining_request_is_done:
            episode_done = True
        # time.sleep(FRESH_TIME)
        return episode_done

    def update_sim_env(self, action):
        # action_space的最后一位是null
        if action != self.action_space_dimension - 1:
            # 确保 action 有mask 不会选择队列为空的剩余时间队列
            # time_stamp = time.time()
            submit_index = np.random.choice(
                self.active_request_group_by_remaining_time_list[action].
                __len__())
            end_request = list(
                self.active_request_group_by_remaining_time_list[action]
                [submit_index])
            # 把提交的任务从active_request_list中删除
            del self.active_request_group_by_remaining_time_list[action][
                submit_index]
            end_request[WAIT_TIME_INDEX] = t - end_request[ARRIVE_TIME_INDEX]
            self.success_request_list.append(end_request)
            # 更新状态
            self.active_request_group_by_remaining_time_list_to_state()
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

    # request_list -> state
    def active_request_group_by_remaining_time_list_to_state(self):
        state = []
        for active_request_group_by_remaining_time in self.active_request_group_by_remaining_time_list:
            state.append(len(active_request_group_by_remaining_time))
        state.append(THRESHOLD)
        self.state_record = state

    def is_correct(self):
        all_request_id_list = []
        for request_in_dic in self.new_arrive_request_in_dic:
            all_request_id_list.append(request_in_dic[REQUEST_ID_INDEX])
        all_request_after_episode_list = []
        all_request_id_after_episode_list = []
        for request in self.fail_request_list:
            all_request_after_episode_list.append(request)
            all_request_id_after_episode_list.append(request[REQUEST_ID_INDEX])
        for request in self.success_request_list:
            all_request_after_episode_list.append(request)
            all_request_id_after_episode_list.append(request[REQUEST_ID_INDEX])
        all_request_id_list.sort()
        all_request_id_after_episode_list.sort()
        return all_request_id_after_episode_list == all_request_id_list

    def get_success_rate(self):
        all_request = []
        for request_in_dic in self.new_arrive_request_in_dic:
            all_request.append(request_in_dic[REQUEST_ID_INDEX])
        all_request_num = all_request.__len__()
        return self.success_request_list.__len__() / all_request_num

    def save_success_request(self):
        # success_request_list[request_id, arrive_time, rtl, wait_time]
        headers = ['request_id', 'arrive_time', 'rtl', 'wait_time']
        with open(self.end_request_result_path + 'success_request_list' +
                  str(self.episode) + '.csv',
                  'w',
                  newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(self.success_request_list)

    # 现在用t来表示，真实环境中收集[t-1,t)到来的请求直接给出
    def get_new_arrive_request_list(self):
        now_new_arrive_request_list = []
        for i in range(STATE_DIMENSION - 1):
            now_new_arrive_request_list.append([])
        if t in self.arriveTime_request_dic:
            for request_in_dic in self.arriveTime_request_dic[t]:
                # request_in_dic的形式为[request_id, arrive_time, rtl]
                # request [request_id, arrive_time, rtl, remaining_time]
                # request_in_dic 转为request
                request = list(request_in_dic)
                # 刚加进缓冲时 remaining_time=rtl
                request.append(request_in_dic[RTL_INDEX])
                now_new_arrive_request_list[request[REMAINING_TIME_INDEX] //
                                            TIME_UNIT_IN_ON_SECOND].append(
                                                request)
        return now_new_arrive_request_list

    #   初始状态
    def reset(self, episode):
        self.episode = episode
        self.simulate_time = 0
        t_to_zero()
        self.success_request_list = []
        self.fail_request_list = []
        self.active_request_group_by_remaining_time_list = self.get_new_arrive_request_list(
        )
        self.active_request_group_by_remaining_time_list_to_state()
        return self.state_record

    def get_active_request_sum(self):
        sum = 0
        for active_request in self.active_request_group_by_remaining_time_list:
            sum += active_request.__len__()
        return sum

    def get_not_avail_actions(self):
        not_avail_actions = []
        for i in range(len(self.state_record) - 1):
            if self.state_record[i] == 0:
                not_avail_actions.append(i)
        return not_avail_actions


env = RequestEnv()
