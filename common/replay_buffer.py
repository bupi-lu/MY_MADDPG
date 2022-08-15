import sys
import threading
import numpy as np
import random

class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock: #线程锁是一个执行完再执行下一个，如果不加就会同时进行 道子buffer存储错误
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]

    def samplebyMyWayFuncofDDPG(self, buf, batch_size):
        res = np.zeros(buf.shape)
        for i in range(batch_size):
            if i == batch_size-1:
                res[i] = buf[0]
            else:
                res[i] = buf[i+1]
        return res

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        count = 1
        for key in self.buffer.keys():
            if count % 4 != 0:
                temp_buffer[key] = self.buffer[key][idx]
            else:
                temp_buffer[key] = self.samplebyMyWayFuncofDDPG(self.buffer["o_%d" % (count//4 - 1)][idx],batch_size)
            count += 1
        
        return temp_buffer

    def _get_storage_idx(self, inc=None):  
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
