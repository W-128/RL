import sys
import os
from request_env import RequestEnv
from DQN.dqn import DQN
from train_test import train, test
import datetime
from my_common.utils import plot_rewards, plot_rewards_cn
from my_common.utils import save_results, make_dir
from my_common.utils import save_success_rate, plot_success_rate
import torch

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    '''超参数
    '''

    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DQN'  # 算法名称
        self.env_name = 'request_env'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层
        ################################################################################

        ################################# 保存结果相关参数 ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def env_agent_config(cfg):
    '''创建环境和智能体
    Args:
        cfg ([type]): [description]
        seed (int, optional): 随机种子. Defaults to 1.
    Returns:
        env [type]: 环境
        agent : 智能体
    '''
    env = RequestEnv()
    state_dim = env.state_dimension  # 状态维度
    action_dim = env.action_space_dimension  # 动作空间维度
    agent = DQN(state_dim, action_dim, cfg)  # 创建智能体
    return env, agent


cfg = Config()
# 训练
env, agent = env_agent_config(cfg)
rewards, ma_rewards, success_rate = train(cfg, env, agent)
make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
agent.save(path=cfg.model_path)  # 保存模型
save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)  # 保存结果
plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
plot_success_rate(success_rate, cfg, tag="train")
# 测试
env, agent = env_agent_config(cfg)
agent.load(path=cfg.model_path)  # 导入模型
rewards, ma_rewards, success_rate = test(cfg, env, agent)
save_results(rewards, ma_rewards, tag='test', path=cfg.result_path)  # 保存结果
plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果
plot_success_rate(success_rate, cfg, tag="test")
