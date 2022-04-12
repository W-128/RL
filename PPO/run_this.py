import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath + '/my_common')
sys.path.append(rootPath + '/my_common/utils')

from PPO.eRL_demo_PPOinSingleFile import demo_continuous_action, evaluate_agent

demo_continuous_action()
