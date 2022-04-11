import time


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    success_rate = []
    for i_ep in range(cfg.train_eps):
        start_time = time.time()
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset(i_ep)  # 重置环境,即开始新的回合
        while True:
            # debug
            not_avail_action = env.get_not_avail_actions()
            # print('observation:' + str(state))
            # print('not_available_action' + str(not_avail_action))
            action = agent.choose_action(state)  # 根据算法选择一个动作
            # print('action:' + str(action))
            next_state, reward, done = env.step(action)  # 与环境进行一次动作交互
            agent.learn(state, action, reward, next_state, done)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        success_rate.append(env.get_success_rate())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        end_time = time.time()
        print("回合数：{}/{}，奖励{:.1f}，耗时{:.1f}s".format(i_ep + 1, cfg.train_eps, ep_reward, end_time - start_time))
    print('完成训练！')
    return rewards, ma_rewards, success_rate


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}')
    for item in agent.q_table.items():
        print(item)
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 滑动平均的奖励
    success_rate = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset(i_ep)  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            # debug
            not_avail_action = env.get_not_avail_actions()
            # print('observation:' + str(state))
            # print('not_available_action' + str(not_avail_action))
            action = agent.predict(state, not_avail_action)  # 根据算法选择一个动作
            next_state, reward, done = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            # print('action:' + str(action))
            if done:
                env.save_success_request()
                break
        success_rate.append(env.get_success_rate())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards, success_rate
