import time

class Logger:
    def __init__(self):
        self.start_times = []
        self.block_names = []

    def get_inside(self, block_name):
        print('| '*len(self.block_names)+'|>> '+block_name+' started.')
        self.block_names.append(block_name)
        self.start_times.append(time.time())

    def get_out(self):
        print('| '*(len(self.block_names)-1)+'<<< '+self.block_names.pop()+' ended after %.5s seconds.'%(time.time() - self.start_times.pop()))


LOGGER = Logger()


def timing_logger(func):
    global LOGGER
    def wrapped(*args, **kwargs):
        LOGGER.get_inside(func.__name__)
        result = func(*args, **kwargs)
        LOGGER.get_out()
        return result
    return wrapped