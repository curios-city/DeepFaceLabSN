import multiprocessing
from core.interact import interact as io

class MPFunc():
    def __init__(self, func):
        self.func = func
        
        self.s2c = multiprocessing.Queue()
        self.c2s = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        
        io.add_process_messages_callback(self.io_callback)

    def io_callback(self):        
        while not self.c2s.empty():
            func_args, func_kwargs = self.c2s.get()
            self.s2c.put ( self.func (*func_args, **func_kwargs) )

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.c2s.put ( (args, kwargs) )
            return self.s2c.get()

    def __getstate__(self):
        return {'s2c':self.s2c, 'c2s':self.c2s, 'lock':self.lock}
