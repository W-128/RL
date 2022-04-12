from my_common.utils import concurrent_request_num_per_second_list_to_concurrent_request_num

# t=1000ms
TIME_UNIT = 1
TIME_UNIT_IN_ON_SECOND = int(1 / TIME_UNIT)
# 生成一个先升后降的流量 时长320s 0to80to0 阈值40
header = ['concurrent_request_num_per_second']


def request_number(second):
    if int(0.5 * second) == 0:
        return 1
    else:
        return int(0.5 * second)


temp_list = []
for second in range(160):
    temp_list.append(request_number(second))

concurrent_request_num_per_second = list(temp_list)
temp_list.reverse()
concurrent_request_num_per_second = concurrent_request_num_per_second + temp_list
# concurrent_request_num_per_second index是到达时间 element是请求个数
print(concurrent_request_num_per_second)

concurrent_request_num_per_second_list_to_concurrent_request_num(
    concurrent_request_num_per_second)

# # 先造只有rtl1 和rtl3
# rtl_list = [1, 3]
# for i in range(len(rtl_list)):
#     rtl_list[i] = rtl_list[i] * TIME_UNIT_IN_ON_SECOND
#
# request_list = []
# for i in range(len(concurrent_request_num_per_second)):
#     request_sum_the_second = concurrent_request_num_per_second[i]
#     request_num_in_the_second = [int(request_sum_the_second * TIME_UNIT)] * (TIME_UNIT_IN_ON_SECOND)
#     remain_request_num = request_sum_the_second - int(request_sum_the_second * TIME_UNIT) * (TIME_UNIT_IN_ON_SECOND)
#     for j in range(remain_request_num):
#         request_num_in_the_second[j] = request_num_in_the_second[j] + 1
#     np.random.shuffle(request_num_in_the_second)
#     print(request_num_in_the_second)
#     for k in range(TIME_UNIT_IN_ON_SECOND):
#         for h in range(request_num_in_the_second[k]):
#             # [request_id, arrive_time, rtl]
#             request = []
#             request.append(str(uuid.uuid1()))
#             request.append(i * TIME_UNIT_IN_ON_SECOND + k)
#             request.append(np.random.choice(rtl_list))
#             request_list.append(request)
#
# headers = ['request_id', 'arrive_time', 'rtl']
# with open('concurrent_request_num.csv', 'w', newline='')as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(headers)
#     f_csv.writerows(request_list)
