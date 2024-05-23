from .MPSharedList import MPSharedList
import multiprocessing
import threading
import time

import numpy as np


class IndexHost():
    """
    Provides random shuffled indexes for multiprocesses
    """
    def __init__(self, indexes_count, rnd_seed=None):
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []
        self.thread = threading.Thread(target=self.host_thread, args=(indexes_count,rnd_seed) )
        self.thread.daemon = True
        self.thread.start()

    def host_thread(self, indexes_count, rnd_seed):
        rnd_state = np.random.RandomState(rnd_seed) if rnd_seed is not None else np.random

        idxs = [*range(indexes_count)]
        shuffle_idxs = []
        sq = self.sq

        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, count = obj[0], obj[1]

                result = []
                for i in range(count):
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = idxs.copy()
                        rnd_state.shuffle(shuffle_idxs)
                    result.append(shuffle_idxs.pop())
                self.cqs[cq_id].put (result)

            time.sleep(0.001)

    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq )
        cq_id = len(self.cqs)-1
        return IndexHost.Cli(self.sq, cq, cq_id)

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def multi_get(self, count):
            self.sq.put ( (self.cq_id,count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

class Index2DHost():
    """
    Provides random shuffled indexes for multiprocesses
    """
    def __init__(self, indexes2D):
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []
        self.thread = threading.Thread(target=self.host_thread, args=(indexes2D,) )
        self.thread.daemon = True
        self.thread.start()

    def host_thread(self, indexes2D):
        indexes2D_len = len(indexes2D)

        idxs = [*range(indexes2D_len)]
        idxs_2D = [None]*indexes2D_len
        shuffle_idxs = []
        shuffle_idxs_2D = [None]*indexes2D_len
        for i in range(indexes2D_len):
            idxs_2D[i] = [*range(len(indexes2D[i]))]
            shuffle_idxs_2D[i] = []

        #print(idxs)
        #print(idxs_2D)
        sq = self.sq

        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, count = obj[0], obj[1]

                result = []
                for i in range(count):
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = idxs.copy()
                        np.random.shuffle(shuffle_idxs)

                    idx_1D = shuffle_idxs.pop()
                    
                    #print(f'idx_1D = {idx_1D}, len(shuffle_idxs_2D[idx_1D])= {len(shuffle_idxs_2D[idx_1D])}')
                    
                    if len(shuffle_idxs_2D[idx_1D]) == 0:
                        shuffle_idxs_2D[idx_1D] = idxs_2D[idx_1D].copy()
                        #print(f'new shuffle_idxs_2d for {idx_1D} = { shuffle_idxs_2D[idx_1D] }')
                        
                        #print(f'len(shuffle_idxs_2D[idx_1D])= {len(shuffle_idxs_2D[idx_1D])}')
                    
                        np.random.shuffle( shuffle_idxs_2D[idx_1D] )

                    idx_2D = shuffle_idxs_2D[idx_1D].pop()
                    
                    #print(f'len(shuffle_idxs_2D[idx_1D])= {len(shuffle_idxs_2D[idx_1D])}')
                    
                    #print(f'idx_2D = {idx_2D}')
                    

                    result.append( indexes2D[idx_1D][idx_2D])

                self.cqs[cq_id].put (result)

            time.sleep(0.001)

    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq )
        cq_id = len(self.cqs)-1
        return Index2DHost.Cli(self.sq, cq, cq_id)

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def multi_get(self, count):
            self.sq.put ( (self.cq_id,count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

class ListHost():
    def __init__(self, list_):
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []
        self.m_list = list_
        self.thread = threading.Thread(target=self.host_thread)
        self.thread.daemon = True
        self.thread.start()

    def host_thread(self):
        sq = self.sq
        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, cmd = obj[0], obj[1]

                if cmd == 0:
                    self.cqs[cq_id].put ( len(self.m_list) )
                elif cmd == 1:
                    idx = obj[2]
                    item = self.m_list[idx ]
                    self.cqs[cq_id].put ( item )
                elif cmd == 2:
                    result = []
                    for item in obj[2]:
                        result.append ( self.m_list[item] )
                    self.cqs[cq_id].put ( result )
                elif cmd == 3:
                    self.m_list.insert(obj[2], obj[3])
                elif cmd == 4:
                    self.m_list.append(obj[2])
                elif cmd == 5:
                    self.m_list.extend(obj[2])

            time.sleep(0.005)

    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq )
        cq_id = len(self.cqs)-1
        return ListHost.Cli(self.sq, cq, cq_id)

    def get_list(self):
        return self.list_

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def __len__(self):
            self.sq.put ( (self.cq_id,0) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

        def __getitem__(self, key):
            self.sq.put ( (self.cq_id,1,key) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

        def multi_get(self, keys):
            self.sq.put ( (self.cq_id,2,keys) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

        def insert(self, index, item):
            self.sq.put ( (self.cq_id,3,index,item) )

        def append(self, item):
            self.sq.put ( (self.cq_id,4,item) )

        def extend(self, items):
            self.sq.put ( (self.cq_id,5,items) )



class DictHost():
    def __init__(self, d, num_users):
        self.sqs = [ multiprocessing.Queue() for _ in range(num_users) ]
        self.cqs = [ multiprocessing.Queue() for _ in range(num_users) ]

        self.thread = threading.Thread(target=self.host_thread, args=(d,) )
        self.thread.daemon = True
        self.thread.start()

        self.clis = [ DictHostCli(sq,cq) for sq, cq in zip(self.sqs, self.cqs) ]

    def host_thread(self, d):
        while True:
            for sq, cq in zip(self.sqs, self.cqs):
                if not sq.empty():
                    obj = sq.get()
                    cmd = obj[0]
                    if cmd == 0:
                        cq.put (d[ obj[1] ])
                    elif cmd == 1:
                        cq.put ( list(d.keys()) )

            time.sleep(0.005)


    def get_cli(self, n_user):
        return self.clis[n_user]

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

class DictHostCli():
    def __init__(self, sq, cq):
        self.sq = sq
        self.cq = cq

    def __getitem__(self, key):
        self.sq.put ( (0,key) )

        while True:
            if not self.cq.empty():
                return self.cq.get()
            time.sleep(0.001)

    def keys(self):
        self.sq.put ( (1,) )
        while True:
            if not self.cq.empty():
                return self.cq.get()
            time.sleep(0.001)
