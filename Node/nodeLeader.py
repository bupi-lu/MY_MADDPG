import torch
import pickle
import time
import json
import threading
import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

from nodeWorker import CnodeWorker

#### 这个类用于Leader节点保存其他节点的状态
### state: 0:未注册；1正常运行；2挂了
class SingleNode():
    def __init__(self,id ,mac,neighbor,role=None,ip = None,config = None,updateDeadInterval = None) -> None:
        self.id = id
        self.mac = mac
        self.ip = ip
        self.config = config
        self.deadtime = 0
        self.updateDeadInterval = updateDeadInterval
        self.state = 0
        self.neighbor = neighbor
        self.role = role
    def setrole(self,role):
        self.role = role
    def setid(self,id):
        self.id = id
    def setmac(self,mac):
        self.mac = mac
    def setIP(self,ip):
        self.ip = ip
    def setdeadtime(self,deadtime):
        self.deadtime = int(deadtime)
    def isDead(self):
        return time.time()>self.deadtime
    def setState(self,state):
        self.state = state
    def updateDeadTime(self):
        self.deadtime = time.time() + self.updateDeadInterval


class CnodeLeader(CnodeWorker):
    def __init__(self, config) -> None:
        super().__init__(config)
        if config.EnableFL:
            self.flmasterID = self.id
            self.optimizeFLJob = self.scheduler.add_job(self.optimizeFL, \
                    'interval', seconds = config.ddpgOptimiseInterval   )
        self.GenaHighLevelTopo(config) # 生成一个全局拓扑
        self.numofNodes = config.topo['numofNodes']
        self.scheduler.add_job(self.updateNodes, 'interval', seconds = config.checkInterval) ## 维护节点状态
        self.scheduler.add_job(self.notifytoAction, 'interval', seconds = config.silenceInterval) ## 通知节点行动
        # for node in self.topology:
        #     nodeinfo = self.topology[node]
        #     self.flParamsBuf[nodeinfo.id] = None

    ### web更新node参数
    # 发送hello的时候才会设置deadtime
    def updateNodes(self):
        for nodeOBJ in self.topology.values():
            if nodeOBJ.isDead():
                nodeOBJ.setState(2)

    ### 此函数的功能是用于让web节点获取总拓扑然后用于维护  ###
    def GenaHighLevelTopo(self,config):
        # topoTemps = self.nodeenv.getTopology()
        topoTemps = config.topo # 从配置文件中提取拓扑， 因为环境不可能提供拓扑，所以拓扑应该由leader的配置文件提供
        topos = topoTemps['topos'] 
        self.topology = {}
        for nodeid in topoTemps['nodes']:
            nodeinfo = topos[nodeid]
            if(type(nodeid) != int):
                nodeid = int(nodeid)
            self.topology[nodeid] = SingleNode(id=nodeid,mac=None,neighbor=nodeinfo['neighbor'],config=None,updateDeadInterval=config.deadInterval)
            if config.EnableFL:
                if nodeid == self.flmasterID:
                    self.topology[nodeid].setrole(1)
                    self.topology[nodeid].setIP(self.ip)
                else:
                    self.topology[nodeid].setrole(0)
    def itsYourTurn(self, dest, term):
        method = "itsYourTurnFunc"
        payload = {"term": term}
        self.request(dest, method, payload)

    def helloFunc(self, params):
        self.topology[params['id']].updateDeadTime()
        self.topology[params['id']].setState(1)

    def GetNodeList(self,_):
        return self.topology

    def registe(self,params):
        config = params['config']
        id = int(params['id'])
        self.topology[id].setState(1)
        self.topology[id].setIP(params['ip'])
        self.topology[id].updateDeadTime()
        self.topology[id].config=config
        # self.logger.info(self.topology[self.flmasterID].ip)
        return  {
                    'permit':'accept',\
                    'neighbor':self.topology[id].neighbor,\
                    'masterID':self.flmasterID,\
                    'masterIP':self.topology[self.flmasterID].ip
                }


    def notifytoAction(self):
        self.term += 1
        time.sleep(self.config.silenceInterval)
        self.logger.info("Notify to action in {} term".format(self.term))
        for nodeID,node in self.topology.items():
            if nodeID == self.id:
                self.myTurnLock.release()
            else:
                if not node.isDead():
                    self.itsYourTurn((node.ip,self.config.port),self.term)
                else:
                    self.logger.error("NodeID:{} Down!deadtime:{},timeNow:{}".format(nodeID,node.deadtime,time.time()))

    def LeadersOwnJob(self):
        while True:
            self.myTurnLock.acquire()
            self.logger.info("Time to take {} 's action".format(self.term))
            # specstatus = self.nodeMana.getSpecStatusFunc(self.id)                   ## NOTE: 从环境获取状态
            specstatus = torch.randn(1,self.config.numofBand).squeeze().tolist() ## NOTE: 随机一个状态出来
            # self.logger.info("Spec Status:{}".format(specstatus))
            prestate = self.nodeLSTM.predict(specstatus)  ## NOTE: 预测场景信息
            prestateList = prestate.squeeze().tolist()      ## NOTE: 转换预测的场景信息
            # self.logger.info("Pre Status:{}".format(prestateList))
            self.logger.info("Predict state（list）：{}".format(prestateList))
            ## 根据预测的场景信息进行决策
            action = self.nodeDDPG.get_exploration_action(prestate)
            actionList = action.squeeze().tolist()
            # self.logger.info("DDPG's action:{}".format(actionList))
            self.logger.info("DDPG's Actor's action:{}".format(actionList))
            ## action = torch.rand(self.config.numofDevices) * self.config.powerBound[1] * self.config.numofBand
            # self.nodeMana.takeActionFunc(self.id,actionList)
            # reward = self.nodeMana.getCommuRateFunc(self.id)
            reward = 100
            # 保存状态
            self.nodeDDPG.store_transition((prestateList,actionList,reward))
            payload = {
                "id":self.id,
                # "actiondate": str(int(time.time())),
                "actiondate": int(time.time()),
                "info": pickle.dumps((prestateList,actionList,reward)).hex()
            }
            self.nodeblock.createTrans(self.nodeblock.execer,"Spectrum",payload)
            ## self.nodeMana.setNextTermFunc()

    def mainLeader(self):
        self.registerFuncs(self.registe,self.helloFunc,self.downloadModelFunc,self.uploadModelFunc)  ### 要先注册conductor中处理函数
        addr = (self.ip,self.config.port)
        self.nodeNetwork.listenon(addr,self.leaderConducter)  ## 监听地址从config.json配置文件中获取 WARNING: 针对mainLeader没有进程的情况，可以考虑用这个listenon作为进程。
        # nodeMaintainJob = threading.Thread(target=self.nodeMaintain)
        # notifytoActionJob = threading.Thread(target=self.notifytoAction)
        ThreadworkersOwnJob = threading.Thread(target=self.LeadersOwnJob)

        ## 开启节点维护线程、通知线程、和worker自身的线程。
        # nodeMaintainJob.start()
        # notifytoActionJob.start()
        ThreadworkersOwnJob.start()

        ## 开启scheduler
        self.scheduler.start()

        ## 等待所有模块都退出。
        # nodeMaintainJob.join()
        # notifytoActionJob.join()
        ThreadworkersOwnJob.join()
        ## 退出

    def main(self):
        self.mainLeader()
    ### web 主分支
    def leaderConducter(self,msg,addr):
        """ leader 节点处理接入请求的j分支函数，
            params：
                msg：消息
                addr：接入的地址，目前没用上，但是callback函数要用
        """
        data = json.loads(msg)
        method = data['method']
        params = data['params']
        return self.runFunc(method,params)   ## 调用函数名为method的以params为参数的函数

    ##########################################################################
