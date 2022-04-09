import datetime
import sys
import os
from request_env import RequestEnv
from agent import QLearningTable

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'Q-learning-Table'  # 算法名称
env_name = 'RequestEnv'


class QlearningConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        # self.device = device  # 检测GPU
        self.train_eps = 400  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        self.gamma = 0.9  # reward的衰减率
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.lr = 0.1  # 学习率


class PlotConfig:
    ''' 绘图相关参数设置
    '''

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
    env = RequestEnv()
    # state_dim = env.observation_space.n  # 状态维度
    # action_dim = env.action_list.n  # 动作维度
    agent = QLearningTable()
    return env, agent


# cfg = DNQConfig()
# plot_cfg = PlotConfig()
# # 训练
# env, agent = env_agent_config(cfg, seed=1)
# rewards, ma_rewards = train(cfg, env, agent)
# make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
# agent.save(path=plot_cfg.model_path)  # 保存模型
# save_results(rewards, ma_rewards, tag='train',
#              path=plot_cfg.result_path)  # 保存结果
# plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
# # 测试
# env, agent = env_agent_config(cfg, seed=10)
# agent.load(path=plot_cfg.model_path)  # 导入模型
# rewards, ma_rewards = test(cfg, env, agent)
# save_results(rewards, ma_rewards, tag='test', path=plot_cfg.result_path)  # 保存结果
# plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果

if __name__ == "__main__":
    env = RequestEnv()
    action_list=list(range(env.n_actions))
    agent = QLearningTable(actions=list(range(env.n_actions)))


