import logging
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
from Network import network,myrpc 
class node():
    def __init__(self,config) -> None:
        self.logger = logging.getLogger("specmanager.{}".format(config.nodeRole))
        self.mac = network.getMACAddressfromENV()
        self.id = int(os.getenv("id"))
        # self.ip = network.getIPfromENV()
        # self.ip = config.ip
        self.ip = network.getIPfromENV()
        self.nodeNetwork = network.nodeNetwork()
        self.request = myrpc.myrpc(config).request 
        self.waitTime = config.requireWaitInterval

        self.overtimeInterval = config.overtimeInterval
        self.keepAliveInterval = config.keepAliveInterval
        self.AggreInterval = config.aggreInterval
        self.config = config
        self.config.mac = self.mac
        self.config.id = self.id

        self.config.ip = self.ip
        self.funcmap = {}
        self.TempDict = {
            'method':None,
            'params':None
        }
    def getIDfromMAC(self, mac):
        return int(mac[-1])

    ############# 注册函数，所有注册的函数将用于客户端请求时调用  #################
    def registerFuncs(self,*funcs):
        for func in funcs:
            self.funcmap[func.__name__] = func
    def runFunc(self,method,params):
        return self.funcmap[method](params)
    #########################################################################
