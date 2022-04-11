import os
from request_env_no_sim import RequestEnvNoSim
from agent import RandomChoose, EDF, EDFSubmitThreshold
from train_test import test
import datetime
import torch
from common.utils import plot_waiting_time_and_require_time
from common.utils import make_dir

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'Q-learning'  # 算法名称
env_name = 'request_env_no_sim'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU


class RandomChooseConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = 'random_choose'  # 算法名称
        self.env_name = env_name  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + self.algo_name + '/results/'  # 保存结果的路径
        self.save = True  # 是否保存图片


class EDFSubmitThresholdConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = 'edf_submit_threshold'  # 算法名称
        self.env_name = env_name  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + self.algo_name + '/results/'  # 保存结果的路径
        self.save = True  # 是否保存图片


class EDFConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = 'edf'  # 算法名称
        self.env_name = env_name  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + self.algo_name + '/results/'  # 保存结果的路径
        self.save = True  # 是否保存图片


env = RequestEnvNoSim()

random_choose_cfg = RandomChooseConfig()
make_dir(random_choose_cfg.result_path)  # 创建模型路径的文件夹
agent = RandomChoose(env.action_dimension)
success_request, waiting_time_index, rtl_index = test(random_choose_cfg, env, agent)
plot_waiting_time_and_require_time(success_request, waiting_time_index, rtl_index, random_choose_cfg)

edf_config = EDFConfig()
make_dir(edf_config.result_path)  # 创建模型路径的文件夹
agent = EDF(env.action_dimension)
success_request, waiting_time_index, rtl_index = test(edf_config, env, agent)
plot_waiting_time_and_require_time(success_request, waiting_time_index, rtl_index, edf_config)

edf_submit_threshold_config = EDFSubmitThresholdConfig()
make_dir(edf_submit_threshold_config.result_path)  # 创建模型路径的文件夹
agent = EDFSubmitThreshold(env.action_dimension)
success_request, waiting_time_index, rtl_index = test(edf_submit_threshold_config, env, agent)
plot_waiting_time_and_require_time(success_request, waiting_time_index, rtl_index, edf_submit_threshold_config)
