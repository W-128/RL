def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}')
    ep_reward = 0  # 记录每个episode的reward
    state = env.reset(0)  # 重置环境, 重新开一局（即开始新的一个回合）
    while True:
        action = agent.predict(state, env.threshold)  # 根据算法选择一个动作
        next_state, reward, done = env.step(action)  # 与环境进行一个交互
        state = next_state  # 更新状态
        ep_reward += reward
        # print('action:' + str(action))
        if done:
            # env.save_success_request()
            break
    print('该回合奖励：' + str(ep_reward))
    print('成功率：{:.1f}%'.format(env.get_success_rate() * 100))
    print('超供量：{:.1f}'.format(env.get_more_provision()))
    print('完成测试！')
