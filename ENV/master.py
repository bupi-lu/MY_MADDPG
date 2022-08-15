import torch

class CmasterAction:
    """description"""
    def __init__(self,kargs):
        self.stateList = None
        self.stateLength = None
        self.numOfSpectrum = kargs.numofBand
        self.powerBound = kargs.powerBound
    def getState(self, slot):
        """ 这个函数返回的是在每个slot  masterAP对信道的占用情况
        """
        return torch.rand(self.numOfSpectrum) * (self.powerBound[1] - self.powerBound[0]) + self.powerBound[0]
    def geneStateList(self):
        pass

