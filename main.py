from ast import arg
from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import argparse
import json
import sys
'''# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="MADDPG")
# #继承默认的一个父类，修改description的描述
# ##### 从json文件读取配置 ######
# parser.add_argument('-f','--configFile',type=str,help='从配置文件读取配置')


## parser.add_argument('--requireWaitInterval',type=float,help="设置每次请求的超时间隔",default=0.5)
## parser.add_argument('--refusedInterval',type=float,help='注册被拒绝之后的等待间隔',default=10)
## parser.add_argument('--overtimeInterval',type=float,help='超时重试等待间隔',default=10)
## parser.add_argument('--keepaliveInterval',type=float,help="保活间隔",default=5)
## parser.add_argument('-ai','--aggreInterval',type=float,help="联邦学习聚合的间隔",default="10")
## parser.add_argument('-acc','--walletAccount',type=str,default="Devin",help="用于创建交易的用户")
## parser.add_argument('-p','--walletPasswd',type=str,default="SDX1998ding",help="区块链钱包的密码")
## parser.add_argument('-ml','--maxCheckLength',type=int,default=10,help="查询区块链的最长的长度")
## parser.add_argument('-di','--deadInterval',type=int,default="300",help="设置节点的保活时间，单位s")
## parser.add_argument('-ci','--cheakInterval',type=float,default="0.1",help="设置web节点检查活性的间隔")

#### 配置模块##################
# args = parser.parse_args()
# configdict = args.__dict__
# #python main.py -f config/configEnv.json
# if configdict['configFile'] is not None:
#     print("loading config:{}...".format(configdict['configFile']))
#     ## configPath = "../config{}.json".format(configdict['nodeRole'])

#     with open(configdict['configFile'],'r') as f:
#         argsFile = json.load(f)
#     for item in argsFile:
#         configdict[item] = argsFile[item]

# class Dict2Class(object):
#     def __init__(self, my_dict):
#         for key in my_dict:
#             setattr(self, key, my_dict[key])
            

# config = Dict2Class(configdict)'''#把这一串放到get_args中 既可以读取config里面的参数 也不改变代码本身的配置参数

if __name__ == '__main__':
    # get the params
    args = get_args()   #设置参数
    env, args = make_env(args)  #初始化环境
    # print(env.APs)
    runner = Runner(args, env)  #初始化agent
    if args.evaluate:
        returns = runner.evaluate() #开始评估了都 应该是训练了之后
        print('Average returns is', returns)
    else:
        runner.run()  #主要跑的这个
