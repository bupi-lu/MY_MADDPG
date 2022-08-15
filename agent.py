import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon): 
        if np.random.uniform() < epsilon: #随机选取动作
            u = np.random.uniform(0, self.args.high_action, self.args.action_shape[self.agent_id])
        else: #如果经验池足够，则开始训练
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to("cuda:0")
            pi = self.policy.actor_network(inputs).squeeze(0).to("cuda:0") #将观测值o输入给actor_network网络
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise  #感觉这里可以改
            u = np.clip(u, 0, self.args.high_action) #归一化
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents) #训练

