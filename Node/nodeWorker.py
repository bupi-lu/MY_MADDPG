import sys
import os
import time
import pickle
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from func_timeout import func_set_timeout,func_timeout,FunctionTimedOut

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
sys.path.append("{}/../".format(path))
# sys.path.append(grpc_path = "{}/../gRPC".format(path))
# sys.path.append("{}/../Block".format(path))
# sys.path.append("{}/../ENV".format(path))
# sys.path.append("{}/../Network".format(path))

####################### 引入需要的模块 #########################
from Block import block
from ENV import envManager as envMana
from LSTM import lstm
from DDPG import ddpg
from Node import node
##############################################


class CnodeWorker(node.node):
    def __init__(self,config) -> None:
        super().__init__(config)
        self.term = 0
        self.myTurnLock = threading.Semaphore(0)
        self.refusedInterval = config.refusedInterval
        ### 场景模块
        self.nodeMana = envMana.envManager(config)
        ### 调度器模块############################## 
        """  定期优化DDPT和LSTM
             定期上传LSTM
             定期聚合LSTM
        """
        self.scheduler = BackgroundScheduler()
        ########################################

        ############## 引入预测网络和决策网络 ###############
        ## smthing about DDPG
        self.nodeDDPG = self.initDDPG(config)
        # registe optimizeDDPT
        self.optimizeDDPTJob = self.scheduler.add_job(self.optimizeDDPT,\
              'interval',seconds=config.lstmOptimiseInterval)
        ## smthing about LSTM
        self.nodeLSTM = self.initLSTM(config)
        # registe optimizeLSTM
        self.optimizeLSTMJob = self.scheduler.add_job(self.optimizeLSTM, \
            'interval',seconds=config.ddpgOptimiseInterval)
        ## smthing about Federal learning
        self.EnableFL = config.EnableFL
        if self.EnableFL:
            self.flmasterID = None  ## 联邦学习主节点的ID
            self.flmasterIP = None
            self.flParamsBuf = {}
            self.theNestestTimeGetingModel = 0
            self.FL_AggreMaxInterval = config.FL_AggreMaxInterval
            self.FL_CommunicatingWay = config.FL_CommunicatingWay
            if config.FL_CommunicatingWay == "BLOCK":
                self.optimizeFLwithBlockJob = self.scheduler.add_job(self.optimizeFLwithBlock,\
                        'interval', seconds = config.ddpgOptimiseInterval)

        ## 区块链相关
        self.nodeblock = block.block(config.walletAccount, config.maxCheckChainLength)
        self.nodeblock.UnlockWalletifNot(config.walletPasswd)
        ## ########################################

    ################################## 初始化DDPG  ###################################
    def initDDPG(self, config):
        state_dim = config.numofBand
        action_dim = config.numofDevices
        action_lim = [config.powerBound[0],config.numofBand*config.powerBound[1]]
        learningRateActor = config.learningRateActor
        learningRateCritic = config.learningRateCritic
        batchSize = config.ddpgBatchSize
        gamma = config.gamma
        tau = config.tau
        ddpgRAMSize = config.ddpgRAMSize
        return ddpg.ddpg(state_dim,action_dim,action_lim,learningRateActor,learningRateCritic,batchSize,ddpgRAMSize,gamma,tau)
    #####################################################################################

    ################################## 初始化LSTM  ################################### 
    def initLSTM(self, config):
        input_size = config.numofBand
        hidden_size = config.numofBand
        out_size = config.numofBand
        learningRateLstm = config.learningRateLstm
        ramSize = config.lstmRAMSize
        batchSize = config.lstmBatchSize
        return lstm.lstm(input_size,hidden_size,out_size,learningRateLstm,batchSize,ramSize)
    #####################################################################################

    ################################## 神经网络optimize job  ###################################  
    def optimizeDDPT(self):
        self.logger.info("Optimize DDPG...")
        self.nodeDDPG.optimize()
        self.logger.info("Optimize DDPG Done.")

    def optimizeLSTM(self):
        self.logger.info("Optimize LSTM...")
        self.nodeLSTM.optimize()
        self.logger.info("Optimize LSTM Dowe.")
        self.uploadModeltoBlock(self.nodeLSTM.load_state_dict_hex()) ## 上链
        if self.EnableFL:
            aggretime = int(time.time())
            if self.FL_CommunicatingWay == 'TCP':
                if self.flmasterID == self.id: ## 聚合节点是自己， 更新到buff中
                    self.uploadModeltoBuff(self.id,aggretime,self.nodeLSTM.load_state_dict_hex())
                else:   ## 自己不是聚合节点， 发送给聚合节点
                    if self.flmasterIP == None:
                        raise Exception("I need a flmasterIP!")
                    else:
                        self.request(self.flmasterSock,"uploadModelFunc",{"id":self.id,"aggretime":aggretime,"paramsHex":self.nodeLSTM.load_state_dict_hex()})
            elif self.FL_CommunicatingWay == 'BLOCK':
                ## 上链  上过了
                if self.flmasterID == self.id:
                    ## 如果自己是聚合节点，不仅需要把模型上链，还需要更新FLbuffer
                    self.uploadModeltoBuff(self.id,aggretime,self.nodeLSTM.load_state_dict_hex())





    def optimizeFL(self):
            if len(self.flParamsBuf) != 0:
                self.logger.info("Aggre the FL params...")
                paramDict = self.FederalLearningAggre()
                if paramDict != {}:
                    paramHEX = pickle.dumps(paramDict).hex()
                    if self.EnableFL:
                        if self.FL_CommunicatingWay == "TCP":
                            for nodeID,nodeObject in self.topology.items():
                                if nodeID == self.id:
                                    self.logger.info("Send params to {}".format(nodeID))
                                    self.downloadModelFunc({"paramsHEX":paramHEX})
                                elif not nodeObject.isDead():
                                    self.logger.info("Send params to {}".format(nodeID))
                                    self.request((nodeObject.ip,self.config.port),"downloadModelFunc",{"paramsHEX":paramHEX})
                        elif self.FL_CommunicatingWay == "BLOCK":
                            self.downloadModelFunc({"paramsHEX": paramHEX}) ## 更新聚合节点自身的网络
                            self.uploadModeltoBlock(paramHEX,id = 0)  ## 将聚合节点上链
            else:
                self.logger.warning("There were no params in FLbuffer")
    def optimizeFLwithBlock(self):
        if self.config.nodeRole == "leader": ## TODO: 这里不应该是leader 但是先按leader写,这里应该判断时不是联邦节点
            ids = set()
            for nodeID, node in self.topology.items():
                if nodeID == self.id:
                    pass
                elif not node.isDead():
                    ids.add(nodeID)
            self.logger.info("Willing to pull id set:{}".format(ids))
            if len(ids) != 0:
                payloads = self.GetTheNewestModelfromBlockofID(ids)
                self.logger.info("Pulled {} params".format(len(payloads)))
                for payload in payloads:
                    id = payload['id']
                    aggretime = payload['aggretime']
                    paramHEX = payload['param']
                    self.uploadModeltoBuff(id,aggretime,paramHEX)
        else:
            payload = self.GetTheNewestModelfromBlockofID()
            if payload != []:
                ## NOTE: 这里可以加超时检查但是先不加
                self.logger.info("Get aggred params succesfully")
                paramHEX = payload['param']
                self.downloadModelFunc({"paramsHEX": paramHEX})
                # self.nodeLSTM.set_state_dict(pickle.loads(bytes.fromhex(paramsHEX)))





    #####################################################################################
    ############################## FL常规模块############################## 
    def uploadModelFunc(self, params):
        """ 
        送进来的param必须是字典变量
        包含了 id，aggretime ， params（bytes）
        """
        id = params['id']
        aggretime = params['aggretime']
        paramsHex = params['paramsHex']
        self.uploadModeltoBuff(id,aggretime,paramsHex)



    def downloadModelFunc(self,params:dict):
        """ 将接收的网络参数加载到LSTM网络
            params->dict: {paramsHEx}

        """
        paramsHEX = params['paramsHEX']
        self.nodeLSTM.set_state_dict(pickle.loads(bytes.fromhex(paramsHEX)))

    def uploadModeltoBuff(self, id, aggretime, paramsHex):
        """ 根据参数更新参数队列中的参数 """
        if id in self.flParamsBuf and self.flParamsBuf[id][0] < aggretime:
            self.flParamsBuf[id] = (aggretime,pickle.loads(bytes.fromhex(paramsHex)))
        elif id not in self.flParamsBuf.keys():
            self.flParamsBuf[id] = (aggretime,pickle.loads(bytes.fromhex(paramsHex)))
        else:
            # self.logger.warning("将Model上传到buff的过程中什么都没有做,具体信息是{},{}".format(id,aggretime))
            self.logger.warning("Doing nothing when uploading the Model to FLBuffer, and the info is: {},{}".format(id,aggretime))

    def FederalLearningAggre(self):
        """ 联邦聚合节点聚合函数 """
        ## params = self.GetModelsfromBlocks()
        timeNow = int(time.time())
        sum_params = {}
        sumIndex = 0
        for nodeid, (aggretime,paramDict) in self.flParamsBuf.items():
            if timeNow - int(aggretime) < self.FL_AggreMaxInterval: 
                sumIndex += 1
                if sum_params == {}:
                    for key,var in paramDict.items():
                        sum_params[key] = var.clone()
                else:
                    for key in sum_params:
                        sum_params[key] = sum_params[key] + paramDict[key] #NOTE 这里可能会有bug
            else:
                self.logger.warning("Nodeid {}'s params is time out, current time:{}, aggretime :{}。".format(nodeid,timeNow,aggretime))
        for key in sum_params:
            sum_params[key] = (sum_params[key] / sumIndex)
        self.logger.info("FederalLearningAggre With {} params.".format(sumIndex))
        return sum_params
    ################################################################################ 
    ################################## 从区块链中获得频谱参数  ###################################

    ### 从区块链获取指定id的最新Model
    ## 返回的值是payload或者payloads数组
    def GetTheNewestModelfromBlockofID(self,id = 0) -> list:
        blocks = self.nodeblock.GetBlocksbyMaxLength()
        payloads = self.nodeblock.GetPayloadinBlocksbyExecer(blocks,self.nodeblock.execer)
        payloadLength = len(payloads)
        payloadindex = payloadLength - 1
        param = []
        if type(id) == int:
            while payloadindex != -1:
                payload = payloads[payloadindex]
                if "model" in payload and payload['model']['id'] == id:
                    param = payload['model']
                    break
                payloadindex = payloadindex - 1
        elif type(id) == set:
            while payloadindex != -1:
                payload = payloads[payloadindex]
                if "model" in payload:
                    if payload['model']['id'] in id:
                        id.remove(payload['model']['id'])
                        param.append(payload['model'])
                        if len(id) == 0:
                            break
                payloadindex = payloadindex - 1
        return param

            

    ### 模型上链
    def uploadModeltoBlock(self,param,id = None):
        """ 上传模型到区块链上
            param: 必须是str 
            id: 
        """
        id = self.id if id == None else id
        # curTime = str(int(time.time()))
        curTime = int(time.time())
        payload = {
            "id": id,
            "aggretime": curTime,
            "param":param
        }
        self.nodeblock.createTrans(self.nodeblock.execer,"Model",payload)
        self.logger.info("upload LSTM model to the blockchain,id:{}".format(payload['id']))
    ####################################################################################################################



