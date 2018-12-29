from datetime import datetime


class CountPerSec:
    def __init__(self):
        self._start_time = None
        self._num_occurencies = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurencies += 1

    def count_per_sec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        if elapsed_time <= 0:
            elapsed_time = 1
        return self._num_occurencies / elapsed_time
