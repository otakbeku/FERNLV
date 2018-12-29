from datetime import datetime


# TODO Too much redundant class and unused method in this project

class CountPerSec:
    """
    I got this from tutorial. I will added the source later. This was used to count FPS from a video
    """

    def __init__(self):
        """
        Initialized the object
        """
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        """
        Start the counting
        :return: self
        """
        self._start_time = datetime.now()
        return self

    def increment(self):
        """
        Increment the number of occurrences
        :return:
        """
        self._num_occurrences += 1

    def count_per_sec(self):
        """
        Giving the FPS number from the number of occurrences
        :return:
        """
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        if elapsed_time <= 0:
            elapsed_time = 1
        return self._num_occurrences / elapsed_time
