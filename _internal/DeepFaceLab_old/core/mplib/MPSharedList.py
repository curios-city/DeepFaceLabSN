import multiprocessing
import pickle
import struct
from core.joblib import Subprocessor

class MPSharedList():
    """
    Provides read-only pickled list of constant objects via shared memory aka 'multiprocessing.Array'
    Thus no 4GB limit for subprocesses.

    supports list concat via + or sum()
    """

    def __init__(self, obj_list):
        if obj_list is None:
            self.obj_counts    = None
            self.table_offsets = None
            self.data_offsets  = None
            self.sh_bs         = None
        else:
            obj_count, table_offset, data_offset, sh_b = MPSharedList.bake_data(obj_list)

            self.obj_counts    = [obj_count]
            self.table_offsets = [table_offset]
            self.data_offsets  = [data_offset]
            self.sh_bs         = [sh_b]

    def __add__(self, o):
        if isinstance(o, MPSharedList):
            m = MPSharedList(None)
            m.obj_counts    = self.obj_counts    + o.obj_counts
            m.table_offsets = self.table_offsets + o.table_offsets
            m.data_offsets  = self.data_offsets  + o.data_offsets
            m.sh_bs         = self.sh_bs         + o.sh_bs
            return m
        elif isinstance(o, int):
            return self
        else:
            raise ValueError(f"MPSharedList object of class {o.__class__} is not supported for __add__ operator.")

    def __radd__(self, o):
        return self+o

    def __len__(self):
        return sum(self.obj_counts)

    def __getitem__(self, key):
        obj_count = sum(self.obj_counts)
        if key < 0:
            key = obj_count+key
        if key < 0 or key >= obj_count:
            raise ValueError("out of range")

        for i in range(len(self.obj_counts)):

            if key < self.obj_counts[i]:
                table_offset = self.table_offsets[i]
                data_offset = self.data_offsets[i]
                sh_b = self.sh_bs[i]
                break
            key -= self.obj_counts[i]

        sh_b = memoryview(sh_b).cast('B')

        offset_start, offset_end = struct.unpack('<QQ', sh_b[ table_offset + key*8     : table_offset + (key+2)*8].tobytes() )

        return pickle.loads( sh_b[ data_offset + offset_start : data_offset + offset_end ].tobytes() )

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    @staticmethod
    def bake_data(obj_list):
        if not isinstance(obj_list, list):
            raise ValueError("MPSharedList: obj_list should be list type.")

        obj_count = len(obj_list)

        if obj_count != 0:
            obj_pickled_ar = [pickle.dumps(o, 4) for o in obj_list]

            table_offset = 0
            table_size   = (obj_count+1)*8
            data_offset  = table_offset + table_size
            data_size    = sum([len(x) for x in obj_pickled_ar])

            sh_b = multiprocessing.RawArray('B', table_size + data_size)
            #sh_b[0:8] = struct.pack('<Q', obj_count)
            sh_b_view = memoryview(sh_b).cast('B')

            offset = 0

            sh_b_table = bytes()
            offsets = []

            offset = 0
            for i in range(obj_count):
                offsets.append(offset)
                offset += len(obj_pickled_ar[i])
            offsets.append(offset)

            sh_b_view[table_offset:table_offset+table_size] = struct.pack( '<'+'Q'*len(offsets), *offsets )

            for i, obj_pickled in enumerate(obj_pickled_ar):
                offset = data_offset+offsets[i]
                sh_b_view[offset:offset+len(obj_pickled)] = obj_pickled_ar[i]

            return obj_count, table_offset, data_offset, sh_b
        return 0, 0, 0, None

