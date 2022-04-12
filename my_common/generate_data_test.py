import pandas as pd
import csv
import uuid
import random

header = ['concurrent_request_num_per_second']

# 生成测试流量
temp_list = list(range(1, 10))
concurrent_request_num = list(temp_list)
temp_list.reverse()
concurrent_request_num = concurrent_request_num + temp_list
# concurrent_request_num_per_second index是到达时间 element是请求个数
print(concurrent_request_num)

# 先造只有rtl1 和rtl3
rtl_list = [1, 3]
request_list = []
for i in range(len(concurrent_request_num)):
    for j in range(concurrent_request_num[i]):
        # [request_id, arrive_time, rtl]
        request = []
        request.append(str(uuid.uuid1()))
        request.append(i)
        if random.random() < 0.5:
            request.append(1)
        else:
            request.append(3)
        request_list.append(request)
headers = ['request_id', 'arrive_time', 'rtl']
with open('concurrent_request_num.csv', 'w', newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(request_list)
