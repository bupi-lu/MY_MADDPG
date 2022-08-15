# from logging import RootLogger
# import os,sys
# path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(path)
# from Network.myrpc import myrpc

# ## leader节点和employee节点均有一个envManager
# class envManager():
#     def __init__(self,config) -> None:
#         self.role = config.nodeRole
#         self.envserveraddress = (config.virtualENVServer['ip'],config.virtualENVServer['port'])
#         self.request = myrpc(config).request

#     def takeActionFunc(self,id,action):
#        params = {"id":id,"action":action}
#        self.request(dest=self.envserveraddress,method="tackActionFunc",params=params,needReturn = True)


#     def getSpecStatusFunc(self, id):
#         """call this function to get the status of the environment

#         :id: 节点ID
#         :returns: 节点坐在位置的状态

#         """
#         return self.request(self.envserveraddress,method="getSpecStatusFunc",params={"id":id},needReturn = True)["specstatus"]

#     def getCommuRateFunc(self,id):
#         """call this function to get the communication rate of
#         "  devices owned by id

#         :id: AP的ID
#         :returnsh

#         """
#         result = self.request(self.envserveraddress,"getCommuRateFunc",{"id":id},needReturn = True)
#         return result["commurate"]
#     def setNextTermFunc(self):
#         self.request(self.envserveraddress,"setNextTermFunc",{})


#     def reset(self):
#         self.request(self.envserveraddress,"reset",{"id":id})
#      #############################################################################
