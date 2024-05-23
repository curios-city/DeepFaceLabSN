import copy
import multiprocessing
import traceback

import cv2
import numpy as np

from core import mplib
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)



class Index2DHost():
    """
    Provides random shuffled 2D indexes for multiprocesses
    """
    def __init__(self, indexes2D):
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []
        self.thread = threading.Thread(target=self.host_thread, args=(indexes2D,) )
        self.thread.daemon = True
        self.thread.start()

    def host_thread(self, indexes2D):
        indexes_counts_len = len(indexes2D)

        idxs = [*range(indexes_counts_len)]
        idxs_2D = [None]*indexes_counts_len
        shuffle_idxs = []
        shuffle_idxs_2D = [None]*indexes_counts_len
        for i in range(indexes_counts_len):
            idxs_2D[i] = indexes2D[i]
            shuffle_idxs_2D[i] = []

        sq = self.sq

        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, cmd = obj[0], obj[1]

                if cmd == 0: #get_1D
                    count = obj[2]

                    result = []
                    for i in range(count):
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = idxs.copy()
                            np.random.shuffle(shuffle_idxs)
                        result.append(shuffle_idxs.pop())
                    self.cqs[cq_id].put (result)
                elif cmd == 1: #get_2D
                    targ_idxs,count = obj[2], obj[3]
                    result = []

                    for targ_idx in targ_idxs:
                        sub_idxs = []
                        for i in range(count):
                            ar = shuffle_idxs_2D[targ_idx]
                            if len(ar) == 0:
                                ar = shuffle_idxs_2D[targ_idx] = idxs_2D[targ_idx].copy()
                                np.random.shuffle(ar)
                            sub_idxs.append(ar.pop())
                        result.append (sub_idxs)
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

        def get_1D(self, count):
            self.sq.put ( (self.cq_id,0, count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

        def get_2D(self, idxs, count):
            self.sq.put ( (self.cq_id,1,idxs,count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)
                
'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFacePerson(SampleGeneratorBase):
    def __init__ (self, samples_path, debug=False, batch_size=1,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        person_id_mode=1,
                        **kwargs):

        super().__init__(debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.person_id_mode = person_id_mode

        raise NotImplementedError("Currently SampleGeneratorFacePerson is not implemented.")

        samples_host = SampleLoader.mp_host (SampleType.FACE, samples_path)
        samples = samples_host.get_list()
        self.samples_len = len(samples)

        if self.samples_len == 0:
            raise ValueError('No training data provided.')

        unique_person_names = { sample.person_name for sample in samples }
        persons_name_idxs = { person_name : [] for person_name in unique_person_names }
        for i,sample in enumerate(samples):
            persons_name_idxs[sample.person_name].append (i)
        indexes2D = [ persons_name_idxs[person_name] for person_name in unique_person_names ]
        index2d_host = Index2DHost(indexes2D)

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (samples_host.create_cli(), index2d_host.create_cli(),) )]
        else:
            self.generators_count = np.clip(multiprocessing.cpu_count(), 2, 4)
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (samples_host.create_cli(), index2d_host.create_cli(),) ) for i in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        samples, index2d_host, = param
        bs = self.batch_size

        while True:
            person_idxs = index2d_host.get_1D(bs)
            samples_idxs = index2d_host.get_2D(person_idxs, 1)

            batches = None
            for n_batch in range(bs):
                person_id = person_idxs[n_batch]
                sample_idx = samples_idxs[n_batch][0]

                sample = samples[ sample_idx ]
                try:
                    x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)
                except:
                    raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]

                    batches += [ [] ]
                    i_person_id = len(batches)-1

                for i in range(len(x)):
                    batches[i].append ( x[i] )

                batches[i_person_id].append ( np.array([person_id]) )

            yield [ np.array(batch) for batch in batches]

    @staticmethod
    def get_person_id_max_count(samples_path):
        return SampleLoader.get_person_id_max_count(samples_path)

"""
if self.person_id_mode==1:
            samples_len = len(samples)
            samples_idxs = [*range(samples_len)]
            shuffle_idxs = []
        elif self.person_id_mode==2:
            persons_count = len(samples)

            person_idxs = []
            for j in range(persons_count):
                for i in range(j+1,persons_count):
                    person_idxs += [ [i,j] ]

            shuffle_person_idxs = []

            samples_idxs = [None]*persons_count
            shuffle_idxs = [None]*persons_count

            for i in range(persons_count):
                samples_idxs[i] = [*range(len(samples[i]))]
                shuffle_idxs[i] = []
        elif self.person_id_mode==3:
            persons_count = len(samples)

            person_idxs = [ *range(persons_count) ]
            shuffle_person_idxs = []

            samples_idxs = [None]*persons_count
            shuffle_idxs = [None]*persons_count

            for i in range(persons_count):
                samples_idxs[i] = [*range(len(samples[i]))]
                shuffle_idxs[i] = []

if self.person_id_mode==2:
                if len(shuffle_person_idxs) == 0:
                    shuffle_person_idxs = person_idxs.copy()
                    np.random.shuffle(shuffle_person_idxs)
                person_ids = shuffle_person_idxs.pop()


            batches = None
            for n_batch in range(self.batch_size):

                if self.person_id_mode==1:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = samples_idxs.copy()
                        np.random.shuffle(shuffle_idxs) ###

                    idx = shuffle_idxs.pop()
                    sample = samples[ idx ]

                    try:
                        x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                    if type(x) != tuple and type(x) != list:
                        raise Exception('SampleProcessor.process returns NOT tuple/list')

                    if batches is None:
                        batches = [ [] for _ in range(len(x)) ]

                        batches += [ [] ]
                        i_person_id = len(batches)-1

                    for i in range(len(x)):
                        batches[i].append ( x[i] )

                    batches[i_person_id].append ( np.array([sample.person_id]) )


                elif self.person_id_mode==2:
                    person_id1, person_id2 = person_ids

                    if len(shuffle_idxs[person_id1]) == 0:
                        shuffle_idxs[person_id1] = samples_idxs[person_id1].copy()
                        np.random.shuffle(shuffle_idxs[person_id1])

                    idx = shuffle_idxs[person_id1].pop()
                    sample1 = samples[person_id1][idx]

                    if len(shuffle_idxs[person_id2]) == 0:
                        shuffle_idxs[person_id2] = samples_idxs[person_id2].copy()
                        np.random.shuffle(shuffle_idxs[person_id2])

                    idx = shuffle_idxs[person_id2].pop()
                    sample2 = samples[person_id2][idx]

                    if sample1 is not None and sample2 is not None:
                        try:
                            x1, = SampleProcessor.process ([sample1], self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample1.filename, traceback.format_exc() ) )

                        try:
                            x2, = SampleProcessor.process ([sample2], self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample2.filename, traceback.format_exc() ) )

                        x1_len = len(x1)
                        if batches is None:
                            batches = [ [] for _ in range(x1_len) ]
                            batches += [ [] ]
                            i_person_id1 = len(batches)-1

                            batches += [ [] for _ in range(len(x2)) ]
                            batches += [ [] ]
                            i_person_id2 = len(batches)-1

                        for i in range(x1_len):
                            batches[i].append ( x1[i] )

                        for i in range(len(x2)):
                            batches[x1_len+1+i].append ( x2[i] )

                        batches[i_person_id1].append ( np.array([sample1.person_id]) )

                        batches[i_person_id2].append ( np.array([sample2.person_id]) )

                elif self.person_id_mode==3:
                    if len(shuffle_person_idxs) == 0:
                        shuffle_person_idxs = person_idxs.copy()
                        np.random.shuffle(shuffle_person_idxs)
                    person_id = shuffle_person_idxs.pop()

                    if len(shuffle_idxs[person_id]) == 0:
                        shuffle_idxs[person_id] = samples_idxs[person_id].copy()
                        np.random.shuffle(shuffle_idxs[person_id])

                    idx = shuffle_idxs[person_id].pop()
                    sample1 = samples[person_id][idx]

                    if len(shuffle_idxs[person_id]) == 0:
                        shuffle_idxs[person_id] = samples_idxs[person_id].copy()
                        np.random.shuffle(shuffle_idxs[person_id])

                    idx = shuffle_idxs[person_id].pop()
                    sample2 = samples[person_id][idx]

                    if sample1 is not None and sample2 is not None:
                        try:
                            x1, = SampleProcessor.process ([sample1], self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample1.filename, traceback.format_exc() ) )

                        try:
                            x2, = SampleProcessor.process ([sample2], self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample2.filename, traceback.format_exc() ) )

                        x1_len = len(x1)
                        if batches is None:
                            batches = [ [] for _ in range(x1_len) ]
                            batches += [ [] ]
                            i_person_id1 = len(batches)-1

                            batches += [ [] for _ in range(len(x2)) ]
                            batches += [ [] ]
                            i_person_id2 = len(batches)-1

                        for i in range(x1_len):
                            batches[i].append ( x1[i] )

                        for i in range(len(x2)):
                            batches[x1_len+1+i].append ( x2[i] )

                        batches[i_person_id1].append ( np.array([sample1.person_id]) )

                        batches[i_person_id2].append ( np.array([sample2.person_id]) )
"""
