import time
import pickle
import json
import nodeWorker
import threading
import torch
class CnodeEmployee(nodeWorker.CnodeWorker):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.leaderSock = (config.leaderip,config.port)
        self.scheduler.add_job(self.hello, 'interval', seconds = config.keepAliveInterval, args = (self.leaderSock,))
        self.registerFuncs(self.itsYourTurnFunc, self.downloadModelFunc,self.uploadModelFunc)

    def itsYourTurnFunc(self,params):
        self.term = params['term']
        self.myTurnLock.release()

    def hello(self,dest):
        """ meta func of saying hello to server """
        self.request(dest,"helloFunc",{'ip':self.ip,'id':self.id})
        # time.sleep(self.config.keepAliveInterval)
    def register(self,configall):
        method = "registe"
        payload = {"ip":self.ip, "id":self.id, "config":configall}
        while True:
            result = self.request(self.leaderSock,method,payload,True)
            if result == False:
                self.logger.error("time out when request")
                time.sleep(self.config.overtimeInterval)
                self.logger.warning("retrying...")
                continue
            if result['permit'] == "accept":
                break
            elif result['permit'] == "refuse":
                time.sleep(self.refusedInterval)
        ## 开启保活
        ## time.sleep(self.config.keepAliveIntrval)
        return (result['masterID'],result['masterIP'])
    def RequestNodeList(self):
        method = "GetNodeList"
        payload={}
        result = self.request(self.leaderSock,method,payload)
        if result == False:
            return None
        else:
            return result["nodes"]
    def workersOwnJob(self):
        while True:
            self.myTurnLock.acquire()
            self.logger.info("Time to take {} 's action".format(self.term))
            ## specstatus = self.nodeMana.getSpecStatusFunc(self.id)                   ## NOTE: 从环境获取状态
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
            ## self.nodeMana.takeActionFunc(self.id,actionList)
            ## reward = self.nodeMana.getCommuRateFunc(self.id)
            reward = 100
            # 保存状态
            self.nodeDDPG.store_transition((prestateList,actionList,reward))
            payload = {
                "id":self.id,
                "actiondate": str(int(time.time())),
                "info": pickle.dumps((prestateList,actionList,reward)).hex()
            }
            self.nodeblock.createTrans(self.nodeblock.execer,"Spectrum",payload)

    def mainWorker(self):
        ## 监听地址处理请求
        self.nodeNetwork.listenon((self.ip,self.config.port),self.workderConductor)
        ## 注册节点
        self.flmasterID,self.flmasterIP = self.register(self.config.__dict__) ##workder 节点向web节点注册
        self.flmasterSock = (self.flmasterIP,self.config.port)
        ## 开启保活
        # self.keepalive(self.leaderSock)
        ## 开启工作流程
        ThreadworkersOwnJob = threading.Thread(target=self.workersOwnJob)
        ThreadworkersOwnJob.start()
        ## 开启各种定时需求
        self.scheduler.start()
        ## threading结束时结束        
        ThreadworkersOwnJob.join()

    def main(self):
        self.mainWorker()
    ### worker主分支
    def workderConductor(self,msg,addr):
        data = json.loads(msg)
        method = data['method']
        params = data['params']
        return self.runFunc(method,params)
