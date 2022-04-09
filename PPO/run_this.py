import sys
import os
from request_env_no_sim import RequestEnvNoSim
from agent import PPO
from train_test import train, test
import datetime
from common.utils import plot_rewards, plot_rewards_cn
from common.utils import save_results, make_dir
from common.utils import save_success_rate, plot_success_rate
import torch

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'PPO'  # 算法名称
env_name = 'request_env'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU


class PPOConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.continuous = True  # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        self.batch_size = 5
        self.gamma = 0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.update_fre = 20  # frequency of agent update


class PlotConfig:
    ''' 绘图相关参数设置'''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


def env_agent_config(cfg):
    '''创建环境和智能体
    Args:
        cfg ([type]): [description]
        seed (int, optional): 随机种子. Defaults to 1.
    Returns:
        env [type]: 环境
        agent : 智能体
    '''
    env = RequestEnvNoSim()
    state_dim = env.state_dimension
    action_dim = env.action_dimension
    agent = PPO(state_dim, action_dim, cfg)
    return env, agent


cfg = PPOConfig()
plot_cfg = PlotConfig()
# 训练
env, agent = env_agent_config(cfg)
rewards, ma_rewards, success_rate = train(cfg, env, agent)
make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
agent.save(path=plot_cfg.model_path)  # 保存模型
save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # 保存结果
plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
plot_success_rate(success_rate, plot_cfg, tag="train")
# 测试
env, agent = env_agent_config(cfg)
agent.load(path=plot_cfg.model_path)  # 导入模型
rewards, ma_rewards, success_rate = test(cfg, env, agent)
save_results(rewards, ma_rewards, tag='test', path=plot_cfg.result_path)  # 保存结果
plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果
plot_success_rate(success_rate, plot_cfg, tag="test")
