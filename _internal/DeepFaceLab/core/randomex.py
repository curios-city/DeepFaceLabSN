import numpy as np

def random_normal( size=(1,), trunc_val = 2.5, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
    len = np.array(size).prod()
    result = np.empty ( (len,) , dtype=np.float32)

    for i in range (len):
        while True:
            x = rnd_state.normal()
            if x >= -trunc_val and x <= trunc_val:
                break
        result[i] = (x / trunc_val)

    return result.reshape ( size )