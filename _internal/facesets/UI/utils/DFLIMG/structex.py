import struct

def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter+fmt_size,) + struct.unpack (fmt, data[counter:counter+fmt_size])
