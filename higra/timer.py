import time
from contextlib import asynccontextmanager

class Timer:
    def __init__(self):
        self.timings = {}

    @asynccontextmanager
    async def track(self, name):
        start = time.time()
        yield
        self.timings[name] = time.time() - start