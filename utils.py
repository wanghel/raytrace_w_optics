

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from intervaltree import Interval, IntervalTree


class ArcIntervalTree(IntervalTree):
    def add_interval(self, beg, end, data):
        if beg > end:
            self[beg-360:end] = data
        else:
            self[beg:end] = data

    def get_intervals(self, point):
        return (self[point], self[point-360])

