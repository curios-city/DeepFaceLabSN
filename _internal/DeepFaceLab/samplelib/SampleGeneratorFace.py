import traceback
import os
import numpy as np

from core import mplib
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)



'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path,
                        pak_name=None,
                        debug=False,
                        batch_size=1,
                        random_ct_samples_path=None,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        uniform_yaw_distribution=False,
                        generators_count=4,
                        ignore_same_path=False,
                        raise_on_no_data=True,                        
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        
        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        samples = SampleLoader.load (SampleType.FACE, samples_path, pak_name=pak_name, ignore_same_path=ignore_same_path)
        self.samples = samples # used to have minimal changes with the repo - used for train analysis 
        self.samples_len = len(samples)
        
        if self.samples_len == 0:
            if raise_on_no_data:
                raise ValueError('未提供训练数据, 请检查aligned文件夹')
            else:
                return
                
        if uniform_yaw_distribution:
            samples_pyr = [ ( idx, sample.get_pitch_yaw_roll() ) for idx, sample in enumerate(samples) ]
            
            grads = 128
            #instead of math.pi / 2, using -1.2,+1.2 because actually maximum yaw for 2DFAN landmarks are -1.2+1.2
            grads_space = np.linspace (-1.2, 1.2,grads)

            yaws_sample_list = [None]*grads
            for g in io.progress_bar_generator ( range(grads), "侧脸排序中"):
                yaw = grads_space[g]
                next_yaw = grads_space[g+1] if g < grads-1 else yaw

                yaw_samples = []
                for idx, pyr in samples_pyr:
                    s_yaw = -pyr[1]
                    if (g == 0          and s_yaw < next_yaw) or \
                    (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
                    (g == grads-1    and s_yaw >= yaw):
                        yaw_samples += [ idx ]
                if len(yaw_samples) > 0:
                    yaws_sample_list[g] = yaw_samples
            
            yaws_sample_list = [ y for y in yaws_sample_list if y is not None ]
            
            index_host = mplib.Index2DHost( yaws_sample_list )
        else:
            index_host = mplib.IndexHost(self.samples_len)

        if random_ct_samples_path is not None:
            ct_samples = SampleLoader.load (SampleType.FACE, random_ct_samples_path, pak_name=pak_name, ignore_same_path=ignore_same_path)
            ct_index_host = mplib.IndexHost( len(ct_samples) )
        else:
            ct_samples = None
            ct_index_host = None

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (samples, index_host.create_cli(), ct_samples, ct_index_host.create_cli() if ct_index_host is not None else None) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (samples, index_host.create_cli(), ct_samples, ct_index_host.create_cli() if ct_index_host is not None else None), start_now=False ) \
                               for i in range(self.generators_count) ]
                               
            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1
        
        self.initialized = True
        
    #overridable
    def is_initialized(self):
        return self.initialized
        
    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            return []
            
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        samples, index_host, ct_samples, ct_index_host = param
 
        bs = self.batch_size

        # needed to know if the sample will be flipped or not
        random_flip = None
        while True:
            batches = None
            filenames = []

            indexes = index_host.multi_get(bs)
            ct_indexes = ct_index_host.multi_get(bs) if ct_samples is not None else None

            for n_batch in range(bs):
                sample_idx = indexes[n_batch]
                sample = samples[sample_idx]

                ct_sample = None
                if ct_samples is not None:
                    ct_sample = ct_samples[ct_indexes[n_batch]]

                try:
                    
                    output_samples, random_flip = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                except:
                    raise Exception ("样本 %s 发生异常. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(output_samples[0])) ]

                # we need only the first element (list) of output_samples list
                for i in range(len(output_samples[0])):
                    batches[i].append(output_samples[0][i])

                if random_flip is not None:
                    filenames.append(f"{os.path.basename(sample.filename)} {'(flipped)' if random_flip else ''}")
                else:
                    filenames.append(os.path.basename(sample.filename))

            yield ([ np.array(batch) for batch in batches], filenames)
