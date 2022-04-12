from request_env_no_sim import RequestEnvNoSim

env = RequestEnvNoSim()
env.active_request_group_by_remaining_time_list = [[0] * 10, [1] * 7, [2] * 1, [3] * 1, [4] * 8, []]

dict={'1':1,'2':[2,3,45]}
print(dict.__len__())
