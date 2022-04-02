import numpy as np
import pandas as pd
from dateutil.parser import parse
import csv
from operator import itemgetter
import time

# 状态向量的维数/单位时间在缓存系统中的最大未执行请求数
STATE_DIMENSION = 4
# 引擎能承受的单位时间最大并发量
THRESHOLD = 2
# 当active_request_list的长度不足STATE_DIMENSION时，state的填充元素
PADDING_ELEMENT = 0
# 实时用的话，这个地方无法事先写好，只能每秒来append
# 现在先 直接从文件读取
# request=[request_id, arrive_time, rtl, remaining_time]
# end_request[request_id, arrive_time, rtl, submit_time]
REQUEST_ID_INDEX = 0
ARRIVE_TIME_INDEX = 1
RTL_INDEX = 2
REMAINING_TIME_INDEX = 3
SUBMIT_TIME_INDEX = 3
NEW_ARRIVE_REQUEST = []

FRESH_TIME = 900

# t
t = 0


def T_add_one():
    global t
    T = T + 1


def T_to_zero():
    global t
    T = 0


filename = 'concurrent_request_num.csv'
data = pd.read_csv(filename, header=0)
for i in range(0, len(data)):
    # [request_id, arrive_time, rtl, remaining_time]
    request = [data.loc[i, 'request_id'], data.loc[i, 'arrive_time'], data.loc[i, 'rtl'], data.loc[i, 'rtl']]
    NEW_ARRIVE_REQUEST.append(request)

arriveTime_request_dic = {}
for request in NEW_ARRIVE_REQUEST:
    if request[ARRIVE_TIME_INDEX] in arriveTime_request_dic:
        arriveTime_request_dic[request[ARRIVE_TIME_INDEX]].append(request)
    else:
        request_list = [request]
        arriveTime_request_dic[request[ARRIVE_TIME_INDEX]] = request_list


# class Request:
#     def __init__(self, request_id, arrive_time, rtl):
#         self.request_id = request_id
#         self.arrive_time = arrive_time
#         self.remaining_time = rtl


class RequestEnv:
    def __init__(self):
        action_space_list = []
        # 每一个action是一个1*STATE_DIMENSION的行向量 共有2^STATE_DIMENSION个动作/行
        for i in range(int(np.exp2(STATE_DIMENSION))):
            action_list = []
            bin_i = bin(i)[2:]
            # 填充到STATE_DIMENSION长度的字符串还需要的长度
            len_padding = STATE_DIMENSION - len(bin_i)

            bin_i = bin(i)[2:]
            bin_i_len = len(bin_i)
            # 填充到STATE_DIMENSION长度的字符串还需要的长度
            len_padding = STATE_DIMENSION - bin_i_len
            while len_padding != 0:
                action_list.append(0)
                len_padding = len_padding - 1
            for j in range(bin_i_len):
                action_list.append(int(bin_i[j]))
            # 排除1出现>THRESHOLD次的动作
            if action_list.count(1) <= THRESHOLD:
                action_space_list.append(action_list)
            # list to array
        self.action_space = np.array(action_space_list)
        self.n_actions = self.action_space.shape[0]
        self.active_request_list = []
        self.end_request_list = []

    # 返回奖励值和下一个状态
    def step(self, action):
        done = False
        active_request_list = self.active_request_list
        new_active_request_list = []
        action_list = self.action_space[action]
        end_request_list = []
        for i in range(len(active_request_list)):
            # action选择该任务执行 or 此刻到期了 要么执行了 要么失败了
            if action_list[i] == 1 or active_request_list[i][REMAINING_TIME_INDEX] == 1:
                # 执行
                if action_list[i] == 1:
                    end_request_list.append(active_request_list[i] + [1])
                # 未执行
                else:
                    end_request_list.append(active_request_list[i] + [-1])
            #  remainingTime--
            else:
                active_request_list[i][REMAINING_TIME_INDEX] = active_request_list[i][REMAINING_TIME_INDEX] - 1
                new_active_request_list.append(active_request_list[i])

        # 环境更新
        done = self.update_env(new_active_request_list)
        # 此时self中的active_request_list已经更新为_{t+1}
        next_state = self.active_request_list_to_state()
        return next_state, self.get_reward(end_request_list), done

        # active_request_group_by_remaining_time_list -> state

    # state的[0:len(active_request_group_by_remaining_time_list)] 和 active_request_group_by_remaining_time_list 一一对应
    def active_request_list_to_state(self):
        state = []
        for active_request in self.active_request_list:
            state.append(active_request[REMAINING_TIME_INDEX])
        while state.__len__() < STATE_DIMENSION:
            state.append(PADDING_ELEMENT)
        return state

    def update_env(self, new_active_request_list):
        episode_done = False
        # 与真实环境交互的话这里需要更改
        new_arrive_request_list = self.get_new_arrive_request_list()
        # 更新active_request 为active_request_{t+1}
        self.active_request_list = new_active_request_list + new_arrive_request_list
        if self.active_request_list.__len__() == 0:
            episode_done = True
        time.sleep(FRESH_TIME)
        return episode_done

    def get_reward(self, end_request_list):
        time_stamp = time.time()
        success_request_sum = 0.0
        fail_request_sum = 0.0
        for request in end_request_list:
            if request[SUBMIT_TIME_INDEX] == -1:
                fail_request_sum = fail_request_sum + 1
            else:
                success_request_sum = success_request_sum + 1
                request[SUBMIT_TIME_INDEX] = time_stamp
            self.end_request_list.append(request)
        reward = success_request_sum / (success_request_sum + fail_request_sum)
        return reward

    # 现在用T来表示，真实环境中收集[t-1,t)到来的请求直接给出
    def get_new_arrive_request_list(self):
        now_new_arrive_request_list = []
        if t in arriveTime_request_dic:
            now_new_arrive_request_list = arriveTime_request_dic[t]
            T_add_one()
        return now_new_arrive_request_list

    #   初始状态
    def reset(self):
        T_to_zero()
        self.active_request_list = self.get_new_arrive_request_list()
        return self.active_request_list_to_state()


#
#     def update_observation(self):
#
#         t=t+1
#
#     def observation_to_state(self):
#         state=[]
#         return state
#


env = RequestEnv()
print(env.reset())
