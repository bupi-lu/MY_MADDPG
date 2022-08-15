### 通过socket接受外来请求函数，但是本文件理论上不提供网络交互只提供环境信息，具体环境与各节点之间的交互应该在Node中实现，

## NOTE：node与每个AP的对应通过mac实现，此处应该还有一个 TODO
### NOTE: 每一个AP的状态（state）应该是对应每个节点的通信速率，reward是总的通信速率
import math
from tkinter import N
import torch
import collections
import master
import sys
import numpy as np
specTuple = collections.namedtuple('specTuple','apindex edindex power')
edapTuple = collections.namedtuple('edapTuple','apindex edindex sinr rate')

def one_hot(num,len):#独热编码
    vec = np.zeros([len],np.int32)
    for i in num:
        vec[i] = 1
    return vec
def one_hot_decorder(num):#解码
    dec = []
    count = 0
    for i in num:
        if i > 0.5 :
            dec.append(count)
        count += 1
    return dec
class CendDevice():
    def __init__(self,coordinate,masterAP,kargs,antennaGain = 1) -> None:
        self.coordinate = coordinate
        self.masterAP = masterAP
        self.antennaGain = antennaGain
        self.args = kargs
        self.power = 0
        self.band = []
    def generate_state(self): #状态生成器
        power = np.random.rand(1)#生成功率的大小 不过这个直接用随机 省的归一化了
        spec_require =np.random.randint(self.args.numofBand,size=np.random.randint(low=1,high=self.args.maxRequestbands),dtype='int') #0 - 12 0表示不需要接入， 1 - 12 表示需要的频段
        return np.hstack((power,one_hot(spec_require,self.args.numofBand))) #状态

class CAP():
    def __init__(self,kargs,apProfit) -> None:
        self.id = apProfit["id"]
        self._bandWidth = kargs.bandWidth
        self._numofBand = kargs.numofBand
        self.numofEndDevice = apProfit["devices"]
        self.coordinate = apProfit["coordinate"]
        self.args = kargs
        self.EndDevices = self.GeneEndDevices(apProfit['devices'])

    def GeneEndDevices(self,devicesCoordinate):
        devices = []
        ## 各ap和终端设备的距离还未考量
        for devCoor in devicesCoordinate:
            devices.append( CendDevice(devCoor,self.id,self.args) )
        return devices
    

## 包括主用户和次用户
## 主用户的信道状态怎么处理
class Cenv():
    def __init__(self,kargs) -> None:
        self.bandwidth = kargs.bandWidth
        self.numofBand = kargs.numofBand
        self.powerBound = kargs.powerBound
        self.L0 = kargs.L0
        self.alpha = kargs.alpha
        self.masterAction = master.CmasterAction(kargs)
        self.APs = self.geneAP(kargs)
        self.whiteNoise =  self.P_noise(26,self.bandwidth*math.pow(10,6)) #######白噪声大概率是带宽
        self.term = 0
        self.args = kargs

    def geneAP(self,kargs):
        apsInConfig = kargs.APs
        aps = {}
        # aps["masterAP"] = apsInConfig["masterAP"]["coordinate"]#暂时不要masterAP
        for sap in apsInConfig["slaveAP"]:
            aps[sap["id"]] = CAP(kargs,sap)
        return aps

    @staticmethod
    def P_noise(T,deitaF):
        return 1.38*math.pow(10,-23)*(273.15+T)*deitaF

    def geneActiveDevices(self,numofDevices,fixActiveEndDevice2Max = False):
        """ 为单个AP生成活动终端的数量
        """
        if fixActiveEndDevice2Max:
            numofActionDevices = numofDevices
            activeIndex = torch.ones(numofDevices)
        else:
             activeIndex= torch.randint(2,(numofDevices,))
             numofActionDevices = torch.count_nonzero(activeIndex)
        return (numofActionDevices,activeIndex)

    def poweronRecive(self,sendpower,antennaGain,distance):
        """ 点对点发送情况下，接收端的信号强度 """
        """ TODO : 这里边有一个 1e-3 我还不知道是什么 """
        return sendpower*1e-3*antennaGain/self.L0/math.pow(distance,self.alpha)
    @staticmethod
    def Shannon(band,sinr):
        """ 香农公式 """
        return band*math.log2(1+sinr)
    @staticmethod
    def eularDistance(point1,point2):
        """ 欧拉距离 """
        # return np.linalg.norm(np.array(point1)-np.array(point2))
        return torch.norm((torch.tensor(point1)-torch.tensor(point2)).double())

    def getSpecStatusFunc(self, id, device):
        """ 获取id AP的device位置的各信道的占用情况 """
        """ 需要有一个变量存储每个AP和各自device之间通信所使用的信道和发送功率 """
        """ 接收点的干扰应该是其他AP对自己的设备的功率 """ #其他AP吗
        specStatus = torch.zeros(self.numofBand)
        
        if device == -1:
            point1 = self.APs[id].coordinate
        else:
            point1 = self.APs[id].EndDevices[device].coordinate
        ### 这里遍历到了masterAP
        for apID in self.APs:
            if apID == "masterAP":
            ## 主用户部分
                point2 = self.APs[apID]
                distance = self.eularDistance(point1,point2)
                masterSpec = self.masterAction.getState(self.term)
                for index,item in enumerate(masterSpec):
                    specStatus[index] += self.poweronRecive(item,1,distance)
            else:
            ## 次用户部分 #其他AP的干扰
                point2 = self.APs[apID].coordinate
                if point2 != point1:
                    distance = self.eularDistance(point1,point2)
                    # for index,dev in enumerate(self.APs[apID].EndDevices):
                    #     if(apID == id and index == device):
                    #         pass
                    #     else:
                    #         #这里的band应该是这个devices使用的那个信道 也就是action 为 1 的那个 power也要改 这个band得想一下
                    for band in self.APs[id].EndDevices[device].band:
                        specStatus[band] += self.poweronRecive(self.APs[id].EndDevices[device].power, 1 , distance)

        return specStatus.tolist()

    def getCommuRateFunc(self, id, action):#加入action
        """
        # 获取AP id的devices的通信速率之和
        # 通信速率节点自己求还是环境给呀，毕竟环境已经给到了当前的干扰了
        # 然后此处就有一个问题，    节点在sensing环境状态的时候是不是就不能发送了

        call this function to get the communication rate of devices owned by id

        :id: AP的ID
        :returns: ID下每一个设备的通信速率

        """
        #action [......] 要先变成 [[],[],[]]的形式 
        count = 0
        action_EndDevices = []
        new_action = []
        for i in action:
            new_action.append(i)
            count += 1
            if count % (1+self.args.numofBand) == 0:
                action_EndDevices.append(new_action)
                new_action = []
        #这里还应该把dev.band和dev.power给整了！！！ dev.band是一个列表
        #对应devices的action提出来了 那就>0的表示1？ <0的表示0？ 试一下吧
        commuRate = 0
        for index,dev in enumerate(self.APs[id].EndDevices):
            dev.power = max((action_EndDevices[index][0]+self.args.high_action) / 2 * self.args.powerBound[1] , self.args.powerBound[0]+0.1)
            dev.band = one_hot_decorder(action_EndDevices[index][1:])#>high/2 为1表示分配该信道 <为0
            specstatus = self.getSpecStatusFunc(id,index)
            for band in dev.band:
                commuRate += self.Shannon(self.bandwidth,self.poweronRecive(dev.power, 1 , self.eularDistance(dev.coordinate,self.APs[id].coordinate)) / specstatus[band])
        return commuRate,action_EndDevices

    # def takeActionFunc(self, id, action):
    #     """这个函数是为了采取行动

    #     :id: TODO
    #     :action: 存储了本id所有节点的神经网络的输出
    #     :returns: TODO

    #     """
    #     numofEndDevice = self.APs[id].numofEndDevice;
    #     if len(action) == numofEndDevice:
    #         return {'code':-1}
    #     for index,dev in enumerate(self.APs[id].EndDevices):
    #         dev.band = int(action[index]/self.powerBound[1]);
    #         dev.power = action[index]-dev.band
    #     return {'code':0}

    def setNextTermFunc(self):
        self.term += 1
    
    def getKey(self,agent_id):
        count = 0
        for key in self.APs.keys():
            if count == agent_id:
                return key
            count += 1
        return "出错"
    def step(self,agent_id,action,state):#进行一步之后应该获得的奖励等等之类的
        #首先看谱效
        #action【【】，【】，【】】
        #这里的agent_id其实是数字 要进行变化得到APs的id值
        # print(state)
        reward = 0
        commurate,action_EndDevices = self.getCommuRateFunc(self.getKey(agent_id),action)

        #宽度
        count = 0 #看占了几个信道#这个要改
        band_count_list = [] #看这几个enddevice一起用了几个信道
        for action_enddevice in action_EndDevices:
            for i in range(len(action_enddevice[1:])):
                if action_enddevice[i+1] > 0.5:
                    action_enddevice[i+1] = 1
                else:
                    action_enddevice[i+1] = 0
            if len(band_count_list) == 0:
                band_count_list=action_enddevice[1:]
            else:
                band_count_list=[x+y for x,y in zip(band_count_list,action_enddevice[1:])]
        #这样我看通过actor_network出来很多0 这个动作设置是不是还有问题
        for i in band_count_list:
            if i >= 1:
                count += 1
        bandwidth = self.args.bandWidth * max(count,1) #bandwidth假设是单个信道的带宽宽度 最差count是0嘛 为了保证除数不为0 用了max
        #谱效
        SpecEfficiency = commurate / bandwidth

        ###用户满意度
        #那应该把state也传进来 new_action  state和action一样的
        #state [......] 要先变成 [[],[],[]]的形式 
        count = 0
        state_EndDevices = []
        new_state = []
        for i in state:
            new_state.append(i)
            count += 1
            if count % (1+self.args.numofBand) == 0:
                state_EndDevices.append(new_state)
                new_state = []
        for action_,state_ in zip(action_EndDevices,state_EndDevices):
            if action_[0] < state_[0]:
                reward += 1
            for i,j in zip(action_[1:],state_[1:]):
                if i == j and i != 0:
                    reward += 0.1
                elif( i!=j ):
                    reward -= 0.1
        return reward+SpecEfficiency
        

    

    def reset(self):
        """ 这个函数要不要实现还不一定 """ #这里要实现 就是给他重新生成状态 其实弄清楚后这个应该就挺好写了 先写吧
        #我看了之前的那个state 也是由array组成的列表 所以这个也这样写吧
        state = []
        for AP in self.APs:
            if AP == "masterAP":
                continue
            else: #reset 用户需求
                singleAP_state = [enddevice.generate_state() for enddevice in self.APs[AP].EndDevices]
                singleAP_state = np.array(singleAP_state)
                state.append(singleAP_state.flatten()) 
        return state#现在state里面存储的是每个AP对应device的需求 还需要转换[[array、array、array...],[],[]] 这个要改

