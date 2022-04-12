import numpy as np
from my_common.get_data import get_arrive_time_request_dic

# request=[request_id, arrive_time, rtl, remaining_time]
# success_request_list[request_id, arrive_time, rtl, wait_time]
REQUEST_ID_INDEX = 0
ARRIVE_TIME_INDEX = 1
RTL_INDEX = 2
REMAINING_TIME_INDEX = 3
WAIT_TIME_INDEX = 3
# t=1000ms
TIME_UNIT = 1
TIME_UNIT_IN_ON_SECOND = int(1 / TIME_UNIT)
# 引擎能承受的单位时间最大并发量
THRESHOLD = int(40 / TIME_UNIT_IN_ON_SECOND)
'''
随机提交Threshold个请求
'''


class RandomChoose:

    def __init__(self):
        self.new_arrive_request_in_dic, self.arriveTime_request_dic = get_arrive_time_request_dic(
            ARRIVE_TIME_INDEX)
        self.t = 0
        self.active_request_list = []
        self.get_new_arrive_request()
        self.success_request_list = []
        self.fail_request_list = []

    def get_new_arrive_request(self):
        if self.t in self.arriveTime_request_dic:
            for request_in_dic in self.arriveTime_request_dic[self.t]:
                # request_in_dic的形式为[request_id, arrive_time, rtl]
                # request [request_id, arrive_time, rtl, remaining_time]
                # request_in_dic 转为request
                # 刚加进缓冲时 remaining_time=rtl
                request = list(request_in_dic)
                request.append(request_in_dic[RTL_INDEX])
                self.active_request_list.append(request)

    def submit_request(self):
        remaining_num = THRESHOLD
        while remaining_num != 0 and len(self.active_request_list) != 0:
            submit_index = np.random.choice(len(self.active_request_list))
            success_request = list(self.active_request_list[submit_index])
            del self.active_request_list[submit_index]
            success_request[
                WAIT_TIME_INDEX] = self.t - success_request[ARRIVE_TIME_INDEX]
            self.success_request_list.append(success_request)
            remaining_num = remaining_num - 1

    def step(self):
        self.submit_request()
        self.t = self.t + 1
        for request in self.active_request_list:
            request[REMAINING_TIME_INDEX] = request[REMAINING_TIME_INDEX] - 1
        for request in self.active_request_list[:]:
            if request[REMAINING_TIME_INDEX] < 0:
                self.fail_request_list.append(list(request))
                self.active_request_list.remove(request)

        # time.sleep(FRESH_TIME)
        self.get_new_arrive_request()
        episode_done = False
        if self.t > np.max(list(self.arriveTime_request_dic.keys())
                           ) and self.active_request_list.__len__() == 0:
            episode_done = True
        if episode_done:
            print('环境正确性：' + str(self.is_correct()))
            print('成功率：{:.1f}%'.format(self.get_success_rate() * 100))
            print('超供率：{:.1f}'.format(self.get_more_provision()))
        return episode_done

    def get_success_rate(self):
        return len(self.success_request_list) / len(
            self.new_arrive_request_in_dic)

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

    def get_more_provision(self):
        more_provision_list = []
        for success_request in self.success_request_list:
            more_provision_list.append(
                (success_request[RTL_INDEX] - success_request[WAIT_TIME_INDEX])
                / success_request[RTL_INDEX])
        return np.sum(more_provision_list)


if __name__ == "__main__":
    random_choose = RandomChoose()
    done = False
    while done != True:
        done = random_choose.step()
