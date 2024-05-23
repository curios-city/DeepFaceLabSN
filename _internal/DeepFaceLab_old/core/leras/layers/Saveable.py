import pickle
from pathlib import Path
from core import pathex
import numpy as np

from core.leras import nn

tf = nn.tf

class Saveable():
    def __init__(self, name=None):
        self.name = name

    #override
    def get_weights(self):
        #return tf tensors that should be initialized/loaded/saved
        return []

    #override
    def get_weights_np(self):
        weights = self.get_weights()
        if len(weights) == 0:
            return []
        return nn.tf_sess.run (weights)

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError ('len of lists mismatch')

        tuples = []
        for w, new_w in zip(weights, new_weights):

            if len(w.shape) != new_w.shape:
                new_w = new_w.reshape(w.shape)

            tuples.append ( (w, new_w) )

        nn.batch_set_value (tuples)

    def save_weights(self, filename, force_dtype=None):
        d = {}
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        name = self.name

        for w in weights:
            w_val = nn.tf_sess.run (w).copy()
            w_name_split = w.name.split('/', 1)
            if name != w_name_split[0]:
                raise Exception("weight first name != Saveable.name")

            if force_dtype is not None:
                w_val = w_val.astype(force_dtype)

            d[ w_name_split[1] ] = w_val

        d_dumped = pickle.dumps (d, 4)
        pathex.write_bytes_safe ( Path(filename), d_dumped )

    def load_weights(self, filename):
        """
        returns True if file exists
        """
        filepath = Path(filename)
        if filepath.exists():
            result = True
            d_dumped = filepath.read_bytes()
            d = pickle.loads(d_dumped)
        else:
            return False

        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        try:
            tuples = []
            for w in weights:
                w_name_split = w.name.split('/')
                if self.name != w_name_split[0]:
                    raise Exception("weight first name != Saveable.name")

                sub_w_name = "/".join(w_name_split[1:])

                w_val = d.get(sub_w_name, None)

                if w_val is None:
                    #io.log_err(f"Weight {w.name} was not loaded from file {filename}")
                    tuples.append ( (w, w.initializer) )
                else:
                    w_val = np.reshape( w_val, w.shape.as_list() )
                    tuples.append ( (w, w_val) )

            nn.batch_set_value(tuples)
        except:
            return False

        return True

    def init_weights(self):
        nn.init_weights(self.get_weights())

nn.Saveable = Saveable
