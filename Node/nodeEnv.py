import json
import torch
import sys
import os
import threading
from queue import Queue

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import node
from  ENV import env_


class nodeEnv(node.node):
    def __init__(self,config) -> None:
        super().__init__(config)
        ##  self.env = env_.env(config)
        self.que = Queue()
        ## self.numofNode = self.env.numofAPs
        self.numofNode = 3
        self.env = env_.Cenv(config)


    def reset(self):
        self.env.reset()

    def tackActionFunc(self,params):
        id = params['id']
        action = params['action']
        return self.env.takeActionFunc(id,action)
        


    def getSpecStatusFunc(self,params):
        id = params['id']
        res = self.env.getSpecStatusFunc(id,-1)
        params = {'specstatus':res}
        return params
    def getCommuRateFunc(self, params):
        id = params['id']
        return {'commurate':self.env.getCommuRateFunc(id)}
    def setNextTermFunc(self, params):
        self.env.setNextTermFunc()

    ## 下边三个函数是ID Index IP三者的对应关系，后边可以尝试在env中表示出来
    def getIndexFromID(self,ID) -> int:
        return 0
    def getIDFromIndex(self,index) -> str:
        return "ip"
    def getIPfromID(self,ID) -> str:
        return 'localhost'

    ## def actionHub(self):
    ##     termNow = -1
    ##     lenNow = 0
    ##     actions = torch.zeros(self.numofNode)
    ##     while True:
    ##         term,id,action = self.que.get()
    ##         if(term == termNow):
    ##             lenNow += 1
    ##             actions[self.getIndexFromID(id)] = action
    ##             if(lenNow == self.numofNode):
    ##                 ## reward = self.env.step(action)
    ##                 reward = 0

    ##                 for index in actions:
    ##                     self.giveReward(self.getIPfromID(self.getIDFromIndex(index)),termNow,reward)
    ##                 termNow += 1
    ##         elif term > termNow:
    ##             termNow = term
    ##             lenNow = 0
    def main(self):
        """ 这里可以直接引入env中的函数但是好像不太好，还是在nodeEnv.py调用env中的函数比较好
        """
        self.registerFuncs(self.reset,self.tackActionFunc,self.getSpecStatusFunc,self.getCommuRateFunc,self.setNextTermFunc)  ### 要先注册conductor中处理函数
        addr = (self.ip,self.config.virtualENVServer['port'])
        self.nodeNetwork.listenon(addr,self.Conductor)  ## 监听地址从config.json配置文件中获取
        ## actionHubThread = threading.Thread(target=self.actionHub)
        ## actionHubThread.start()
    def Conductor(self,msg,addr):
        data = json.loads(msg)
        method = data['method']
        params = data['params']
        return self.runFunc(method,params)   ## 调用函数名为method的以params为参数的函数
