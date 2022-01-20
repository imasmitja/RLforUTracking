from utilities.utilities import transpose_list
import numpy as np
import random
from collections import namedtuple, deque
import pickle
import torch

ALPHA = 0.7 #0.7             # if exp_a = 0, pure uniform random from replay buffer. if esp_a = 1, only uses priorities from replay buffer

class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        
        input_to_buffer = transpose_list(transition)
    
        for item in input_to_buffer:
            self.deque.append(item)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)
    
    def save(self,file):
        """"save deque object to train the agent later on"""
        with open(file, "wb") as f:
            pickle.dump(self.deque, f)
        return
    
    def reload(self,file):
        """"reload the deque object to resume training"""
        with open(file, "rb") as f:
            self.deque = pickle.load(f)
        return

    def __len__(self):
        return len(self.deque)

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    

class ReplayBuffer_SummTree:   # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity, seed):
        self.tree = SumTree(capacity)
        self.min_error = 0.01
        self.alpha = ALPHA
        self.seed = random.seed(seed)
       
    def _getPriority(self, error):
        return (error + self.min_error) ** self.alpha
     
    def push(self, transition, error):
        """push into the buffer"""
        input_to_buffer = transpose_list(transition)
        i = 0
        for item in input_to_buffer:
            p = self._getPriority(error[i])
            self.tree.add(p, item) 
            i += 1
     
    def sample(self, n):
        """sample from the buffer"""
        experiences = []
        indexes = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            experiences.append(data)
            indexes.append(idx)
            
        # transpose list of list
        return transpose_list(experiences), indexes
     
    def update(self, idx, error):
        for i in range(len(idx)):
            prio = self._getPriority(error[i])
            p = np.sqrt(np.dot(prio,prio))
            self.tree.update(idx[i], p)
        return self.tree.total()
        
    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.write


