from time import time
import torch


class Profiler(object):
    def __init__(self, verbose=False) -> None:
        self.timer = dict()
        self.time_log = dict()
        self.enabled = False
        self.verbose = verbose

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def tick(self, name):
        if not self.enabled:
            return
        self.timer[name] = time()
        if name not in self.time_log:
            self.time_log[name] = list()

    def tok(self, name):
        if not self.enabled:
            return
        if name not in self.timer:
            return
        torch.cuda.synchronize()
        elapsed = time() - self.timer[name]
        if self.verbose:
            print(f"{name}: {elapsed*1000:.2f} ms")
        else:
            self.time_log[name].append(elapsed * 1000)
