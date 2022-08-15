from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.maddpg import MADDPG
import sys

class Runner:
    def __init__(self, args, env):
        self.args = args  #配置参数
        self.noise = args.noise_rate #噪声
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()  #初始化agent
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self): #最重要的是 我应该怎么引入env 
        returns = []
        for time_step in tqdm(range(self.args.time_steps)): #tqdm是进度条 可以去掉
            # reset the environment
            # if time_step % self.episode_limit == 0:
            s = self.env.reset() #现在其实每一步都要reset
            u = []
            actions = []
            r = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)#select_action是要改的地方
                    u.append(action)
                    actions.append(action)
                    r.append(self.env.step(agent_id,action,s[agent_id]))#一个AP一个AP的加
            # for i in range(self.args.n_agents, self.args.n_players): #这里应该用不到
            #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            #是没有s_next的 我看师兄写的DDPG里面用的s_next就是下一条经验的s 他先sample出来一批样本 然后做的数据转换
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents])#储存经验用的cpu
            #所以这里储存的也应该是（s,a,r） 在simple那里重新写一个s_next
            
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size) #开始采样训练#这里参考师兄的DDPG

                for agent in self.agents:
                    other_agents = self.agents.copy()  #这是干啥 训练other_agents?
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            r = []
            
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], 0, 0)
                    actions.append(action)
                    r.append(self.env.step(agent_id,action,s[agent_id])) #步进，开始执行actions
            rewards = r[0]
                # s = s_next
            returns.append(rewards)
        print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
