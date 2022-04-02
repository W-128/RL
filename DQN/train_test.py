def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    success_rate = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset(i_ep)  # 重置环境,即开始新的回合
        while True:
            # debug
            not_avail_action = env.get_not_avail_actions()
            # print('observation:' + str(state))
            # print('not_available_action' + str(not_avail_action))
            action = agent.choose_action(state, not_avail_action)  # 根据算法选择一个动作
            # print('action:' + str(action))
            next_state, reward, done = env.step(action)  # 与环境进行一次动作交互
            agent.memory.push(state, action, reward, next_state, done)  # 保存transition
            state = next_state  # 更新状态
            agent.update()  # 更新智能体
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        success_rate.append(env.get_success_rate())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    return rewards, ma_rewards, success_rate


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
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
            action = agent.choose_action(state, not_avail_action)  # 根据算法选择一个动作
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
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards, success_rate
