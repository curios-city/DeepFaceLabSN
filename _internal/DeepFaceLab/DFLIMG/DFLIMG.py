from pathlib import Path

from .DFLJPG import DFLJPG

class DFLIMG():

    @staticmethod
    def load(filepath, loader_func=None):
        if filepath.suffix == '.jpg':
            return DFLJPG.load ( str(filepath), loader_func=loader_func )
        else:
            return None
