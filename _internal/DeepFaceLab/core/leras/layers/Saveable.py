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
        if self.name is None:
            raise Exception("name must be defined.")

        name = self.name
        weights = self.get_weights()

        # Temporary file path
        temp_filename = Path(filename).with_suffix('.tmp')

        with open(temp_filename, 'wb') as f:
            pickle.dump(len(weights), f)  # Dump the number of weights as metadata

            for w in weights:
                w_val = nn.tf_sess.run(w)

                w_name_split = w.name.split('/', 1)
                if name != w_name_split[0]:
                    raise Exception("weight first name != Saveable.name")

                if force_dtype is not None:
                    w_val = w_val.astype(force_dtype)

                pickle.dump({w_name_split[1]: w_val}, f)

        # Use the updated write_bytes_safe
        pathex.write_bytes_safe(Path(filename), temp_filename)

    def load_weights(self, filename):
        filepath = Path(filename)
        if not filepath.exists():
            return False

        if self.name is None:
            raise Exception("name must be defined.")

        with open(filename, 'rb') as f:
            try:
                # Attempt to load the first piece of data, which in the new format is the number of weights
                first_item = pickle.load(f)

                if isinstance(first_item, int):
                    # New format detected: first_item is the number of weights
                    num_weights = first_item
                    tuples = []

                    for _ in range(num_weights):
                        w_dict = pickle.load(f)
                        for w_name, w_val in w_dict.items():
                            full_w_name = f"{self.name}/{w_name}"
                            for w in self.get_weights():
                                if w.name == full_w_name:
                                    w_val = np.reshape(w_val, w.shape.as_list())
                                    tuples.append((w, w_val))
                                    break

                    nn.batch_set_value(tuples)

                else:
                    # Old format detected: first_item is the entire weights dictionary
                    d = first_item
                    weights = self.get_weights()
                    tuples = []

                    for w in weights:
                        w_name_split = w.name.split('/')
                        if self.name != w_name_split[0]:
                            raise Exception("weight first name != Saveable.name")

                        sub_w_name = "/".join(w_name_split[1:])
                        w_val = d.get(sub_w_name, None)

                        if w_val is not None:
                            w_val = np.reshape(w_val, w.shape.as_list())
                            tuples.append((w, w_val))

                    nn.batch_set_value(tuples)

            except pickle.UnpicklingError:
                # Handle the exception if the file format is neither new nor old
                return False

        return True

    def init_weights(self):
        nn.init_weights(self.get_weights())

nn.Saveable = Saveable
