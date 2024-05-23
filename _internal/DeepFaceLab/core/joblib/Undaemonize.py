import multiprocessing


class Undaemonize(object):
    """
    Context Manager to resolve AssertionError: daemonic processes are not allowed to have children
    
    Tested in python 3.8.5
    """

    def __init__(self):
        self.p = multiprocessing.process.current_process()
        if 'daemon' in self.p._config:
            self.daemon_status_set = True
        else:
            self.daemon_status_set = False
        self.daemon_status_value = self.p._config.get('daemon')

    def __enter__(self):
        if self.daemon_status_set:
            del self.p._config['daemon']

    def __exit__(self, type, value, traceback):
        if self.daemon_status_set:
            self.p._config['daemon'] = self.daemon_status_value
