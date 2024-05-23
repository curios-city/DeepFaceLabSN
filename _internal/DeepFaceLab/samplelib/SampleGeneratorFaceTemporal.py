import multiprocessing
import pickle
import time
import traceback

import cv2
import numpy as np

from core import mplib
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)


class SampleGeneratorFaceTemporal(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size,
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        generators_count=2,
                        **kwargs):
        super().__init__(debug, batch_size)

        self.temporal_image_count = temporal_image_count
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = generators_count

        samples = SampleLoader.load (SampleType.FACE_TEMPORAL_SORTED, samples_path)
        samples_len = len(samples)
        if samples_len == 0:
            raise ValueError('No training data provided.')

        mult_max = 1
        l = samples_len - ( (self.temporal_image_count)*mult_max - (mult_max-1)  )
        index_host = mplib.IndexHost(l+1)

        pickled_samples = pickle.dumps(samples, 4)
        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (pickled_samples, index_host.create_cli(),) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (pickled_samples, index_host.create_cli(),) ) for i in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param):
        mult_max = 1
        bs = self.batch_size
        pickled_samples, index_host = param
        samples = pickle.loads(pickled_samples)

        while True:
            batches = None

            indexes = index_host.multi_get(bs)

            for n_batch in range(self.batch_size):
                idx = indexes[n_batch]

                temporal_samples = []
                mult = np.random.randint(mult_max)+1
                for i in range( self.temporal_image_count ):
                    sample = samples[ idx+i*mult ]
                    try:
                        temporal_samples += SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)[0]
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(temporal_samples)) ]

                for i in range(len(temporal_samples)):
                    batches[i].append ( temporal_samples[i] )

            yield [ np.array(batch) for batch in batches]
