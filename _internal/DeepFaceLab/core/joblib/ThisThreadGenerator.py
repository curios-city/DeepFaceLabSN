class ThisThreadGenerator(object):
    def __init__(self, generator_func, user_param=None):
        super().__init__()
        self.generator_func = generator_func
        self.user_param = user_param
        self.initialized = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            self.initialized = True
            self.generator_func = self.generator_func(self.user_param)

        return next(self.generator_func)