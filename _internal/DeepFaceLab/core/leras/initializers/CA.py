import multiprocessing
from core.joblib import Subprocessor
import numpy as np

class CAInitializerSubprocessor(Subprocessor):
    @staticmethod
    def generate(shape, dtype=np.float32, eps_std=0.05):
        """
        Super fast implementation of Convolution Aware Initialization for 4D shapes
        Convolution Aware Initialization https://arxiv.org/abs/1702.06295
        """
        if len(shape) != 4:
            raise ValueError("only shape with rank 4 supported.")

        row, column, stack_size, filters_size = shape

        fan_in = stack_size * (row * column)

        kernel_shape = (row, column)

        kernel_fft_shape = np.fft.rfft2(np.zeros(kernel_shape)).shape

        basis_size = np.prod(kernel_fft_shape)
        if basis_size == 1:
            x = np.random.normal( 0.0, eps_std, (filters_size, stack_size, basis_size) )
        else:
            nbb = stack_size // basis_size + 1
            x = np.random.normal(0.0, 1.0, (filters_size, nbb, basis_size, basis_size))
            x = x + np.transpose(x, (0,1,3,2) ) * (1-np.eye(basis_size))
            u, _, v = np.linalg.svd(x)
            x = np.transpose(u, (0,1,3,2) )
            x = np.reshape(x, (filters_size, -1, basis_size) )
            x = x[:,:stack_size,:]

        x = np.reshape(x, ( (filters_size,stack_size,) + kernel_fft_shape ) )

        x = np.fft.irfft2( x, kernel_shape ) \
            + np.random.normal(0, eps_std, (filters_size,stack_size,)+kernel_shape)

        x = x * np.sqrt( (2/fan_in) / np.var(x) )
        x = np.transpose( x, (2, 3, 1, 0) )
        return x.astype(dtype)

    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, shape, dtype = data
            weights = CAInitializerSubprocessor.generate (shape, dtype)
            return idx, weights

    #override
    def __init__(self, data_list):
        self.data_list = data_list
        self.data_list_idxs = [*range(len(data_list))]
        self.result = [None]*len(data_list)
        super().__init__('CAInitializerSubprocessor', CAInitializerSubprocessor.Cli)

    #override
    def process_info_generator(self):
        for i in range( min(multiprocessing.cpu_count(), len(self.data_list)) ):
            yield 'CPU%d' % (i), {}, {}

    #override
    def get_data(self, host_dict):
        if len (self.data_list_idxs) > 0:
            idx = self.data_list_idxs.pop(0)
            shape, dtype = self.data_list[idx]
            return idx, shape, dtype
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.data_list_idxs.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        idx, weights = result
        self.result[idx] = weights

    #override
    def get_result(self):
        return self.result
