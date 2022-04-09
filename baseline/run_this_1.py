import os
from request_env_no_sim import RequestEnvNoSim
from agent import RandomChoose, EDF, EDFSubmitThreshold
from train_test import test
import datetime
import torch

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


class EDFSubmitThresholdConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = 'edf_submit_threshold'  # 算法名称
        self.env_name = env_name  # 环境名称


class EDFConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = 'edf'  # 算法名称
        self.env_name = env_name  # 环境名称


class EDFSubmitThresholdConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = 'edf_submit_threshold'  # 算法名称
        self.env_name = env_name  # 环境名称


env = RequestEnvNoSim()
random_choose_cfg = RandomChooseConfig()
agent = RandomChoose(env.action_dimension)
test(random_choose_cfg, env, agent)
edf_config = EDFConfig()
agent = EDF(env.action_dimension)
test(edf_config, env, agent)
edf_submit_threshold_config = EDFSubmitThresholdConfig()
agent = EDFSubmitThreshold(env.action_dimension)
test(edf_submit_threshold_config, env, agent)
